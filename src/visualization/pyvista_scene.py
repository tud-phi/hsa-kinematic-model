from contextlib import contextmanager
import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import progressbar
import pyvista as pv
import seaborn as sns
from typing import *
import warnings


import spcs_kinematics.jax_math as jmath
from src.utils.check_freq_activation import check_freq_activation


class PyvistaScene:
    """
    A visualizer for the kinematic configuration of an HSA rod
    """

    def __init__(
        self,
        rod_params: Dict,
        gt_settings: Dict = {},
        hat_settings: Dict = {},
        enable_shadows: bool = False,
        tip_down: bool = False,
        floor_center: Tuple = None,
        floor_size: Tuple = None,
        light_position: Tuple = None,
        **kwargs,
    ):
        self.rod_params = rod_params
        self.L0 = rod_params["printed_length"]

        self.gt_settings, self.hat_settings = gt_settings, hat_settings

        self.enable_shadows = enable_shadows
        if enable_shadows:
            warnings.warn(
                "Shadows are not supported by PyVista / VTK for meshes with opacity"
            )
        self.tip_down = tip_down
        self.floor_center = floor_center if floor_center is not None else (0, 0, 0)
        self.floor_size = floor_size if floor_size is not None else (1.5 * self.L0, 1.5 * self.L0)
        self.light_position = light_position if light_position is not None else (0, 0, 5 * self.L0)

        self.backbone_radius = 0.002

        self.filepath = None
        self.t, self.sample_rate, self.frame_rate = 0.0, 1.0, 1.0

        self.outputs_gt, self.outputs_hat = None, None

        # pyvista plotter
        self.pl = None

        # jax function for rapidly computing the rod mesh
        self.rod_mesh_points_fun = jax.jit(jax.vmap(
            fun=jmath.generate_infinitesimal_cylindrical_mesh_points,
            in_axes=(-1, None, None, None, None),
            out_axes=-1,
        ), static_argnames=("r_resolution", "phi_resolution"))

    def run(
        self,
        T_gt: np.ndarray = None,
        T_hat: np.ndarray = None,
        filepath: str = None,
    ):
        self.draw_scene(
            T_gt=T_gt,
            T_hat=T_hat,
        )

        self.pl.show(auto_close=False)

        if filepath is not None:
            # self.pl.window_size = (2500, 2500)
            self.pl.ren_win.SetOffScreenRendering(True)
            self.pl.save_graphic(filepath)

        self.pl.close()

    @contextmanager
    def movie(self, *args, **kwargs):
        try:
            self._setup_movie(*args, **kwargs)
            yield self
        finally:
            self._finish_movie()

    def _setup_movie(
        self, filepath: str, sample_rate: float = 40, frame_rate: float = 20
    ):
        # this method should be run once at the start when creating a movie
        assert (
            frame_rate <= sample_rate
        ), "frame rate of movie should be less than or equal to sample rate"
        assert (
            sample_rate % frame_rate == 0
        ), "sample rate of movie should be a multiple of frame rate"

        self.filepath, self.sample_rate, self.frame_rate = (
            filepath,
            sample_rate,
            frame_rate,
        )

    def _finish_movie(self):
        self.close()

    def animate(
        self,
        T_gt_ts: np.ndarray = None,
        T_hat_ts: np.ndarray = None,
        **kwargs,
    ):
        """
        Animate a trajectory of the HSA robot in PyVista.
        :param T_gt_ts: (n_t, 4, 4, N) array of ground truth rod poses
        :param T_hat_ts: (n_t, 4, 4, N) array of estimated rod poses
            where n_t is the number of timesteps, n_s is the number of segments,
            and N is the number of points on the backbone spline.
        """
        self._setup_movie(**kwargs)
        print("Rendering movie frames...")
        for time_idx in progressbar.progressbar(range(T_gt_ts.shape[0])):
            self.run_timestep(
                None if T_gt_ts is None else T_gt_ts[time_idx],
                None if T_hat_ts is None else T_hat_ts[time_idx],
            )
        self._finish_movie()

    def run_timestep(
        self,
        T_gt: np.ndarray = None,
        T_hat: np.ndarray = None,
    ):
        if self.t == 0:
            self.draw_scene(
                T_gt=T_gt,
                T_hat=T_hat,
            )

            self.pl.open_movie(self.filepath, framerate=self.frame_rate, quality=9)

            self.pl.show(auto_close=False)

            self.pl.ren_win.SetOffScreenRendering(True)

            self.pl.write_frame()  # Write this frame

        elif check_freq_activation(self.t, 1 / self.frame_rate):
            self.update_scene(
                T_gt=T_gt,
                T_hat=T_hat,
            )

            self.pl.write_frame()  # Write this frame

        self.t = self.t + 1.0 / self.sample_rate

    def draw_scene(
        self,
        T_gt: np.ndarray = None,
        T_hat: np.ndarray = None,
    ):
        # create plotter
        plotter_kwargs = {"window_size": [1600, 1500], "lighting": "none"}
        self.pl = pv.Plotter(**plotter_kwargs)

        if T_gt is not None:
            self.outputs_gt = self.draw_meshes(
                T=T_gt,
                show_backbone=self.gt_settings.get("show_backbone", False),
                show_hsa=self.gt_settings.get("show_hsa", True),
                show_orientation_arrows=self.gt_settings.get("show_orientation_arrows", True),
                num_orientation_arrows=self.gt_settings.get("num_orientation_arrows", 10),
                orientation_arrow_indices=self.gt_settings.get("orientation_arrow_indices", None),
                opacity=self.gt_settings.get("opacity", 1.0),
                orientation_arrows_opacity=self.gt_settings.get("orientation_arrows_opacity", None),
                ambient=self.gt_settings.get("ambient", 1.0),
                diffuse=self.gt_settings.get("diffuse", 0.5),
                specular=self.gt_settings.get("specular", 0.0),
            )
        if T_hat is not None:
            self.outputs_hat = self.draw_meshes(
                T=T_hat,
                show_backbone=self.hat_settings.get("show_backbone", False),
                show_hsa=self.hat_settings.get("show_hsa", True),
                show_orientation_arrows=self.hat_settings.get("show_orientation_arrows", True),
                num_orientation_arrows=self.hat_settings.get("num_orientation_arrows", 10),
                orientation_arrow_indices=self.hat_settings.get("orientation_arrow_indices", None),
                opacity=self.hat_settings.get("opacity", 0.5),
                orientation_arrows_opacity=self.hat_settings.get("orientation_arrows_opacity", None),
                ambient=self.hat_settings.get("ambient", 1.0),
                diffuse=self.hat_settings.get("diffuse", 0.5),
                specular=self.hat_settings.get("specular", 0.0),
            )

        # add light
        light = pv.Light(
            position=self.light_position,
            # position=(0, 1 * self.L0, 0.5 * self.L0),
            show_actor=False,
            positional=True,
            cone_angle=60,
            exponent=20,
            intensity=2,
        )
        self.pl.add_light(light)

        # add coordinate axis at origin of base frame
        # self.pl.add_axes_at_origin()
        marker_args = dict(
            cone_radius=0.6,
            shaft_length=0.7,
            tip_length=0.3,
            ambient=0.5,
            label_size=(0.25, 0.1),
        )
        _ = self.pl.add_axes(line_width=10, marker_args=marker_args, color="black")

        # add floor
        # floor = pl.add_floor(face='-z', opacity=0.5, lighting=True, pad=10.0)
        floor = pv.Plane(
            center=self.floor_center,
            i_size=self.floor_size[0],
            j_size=self.floor_size[1],
            i_resolution=10,
            j_resolution=10,
        )
        self.pl.add_mesh(
            floor,
            ambient=0.0,
            diffuse=0.5,
            specular=0.8,
            color="white",
            opacity=1.0,
        )

        # display settings
        if self.enable_shadows:
            self.pl.enable_shadows()  # add shadows
        self.pl.set_background("white")
        if self.tip_down:
            # target camera orientation: x into the picture, y to the right, z down
            self.pl.camera_position = "xz"
            self.pl.camera.roll = 180  # rotate camera to put tip down
            self.pl.camera.azimuth = 90  # rotate camera around z-axis to put the y-axis to the right
            self.pl.camera.elevation = (
                4.5  # slightly tilt down
            )
        else:
            self.pl.camera_position = "xz"
            self.pl.camera.elevation = (
                18.0  # slightly tilt upwards to look from above onto the robot
            )
            self.pl.camera.azimuth = (
                -90.0
            )  # rotate into (-y, -z) plane with x-axis coming out of the screen

    def update_scene(
        self,
        T_gt: np.ndarray = None,
        T_hat: np.ndarray = None,
    ):
        if T_gt is not None:
            self.update_meshes(
                T=T_gt,
                outputs=self.outputs_gt,
            )
        if T_hat is not None:
            self.update_meshes(
                T=T_hat,
                outputs=self.outputs_hat,
            )

    def draw_meshes(
        self,
        T: np.ndarray,
        show_backbone: bool = False,
        show_hsa: bool = True,
        show_orientation_arrows: bool = True,
        num_orientation_arrows: int = 10,
        orientation_arrow_indices: Sequence = None,
        opacity: float = 1.0,
        ambient: float = 1.0,
        diffuse: float = 0.5,
        specular: float = 0.0,
        orientation_arrows_opacity: float = None,
    ) -> Dict:
        outputs = {}

        colors = sns.dark_palette("blueviolet")

        if show_backbone:
            outputs["backbone_mesh"] = self._generate_rod_mesh(
                T,
                outside_radius=self.backbone_radius,
            )
            outputs["backbone_kwargs"] = dict(
                color=colors[0],
                opacity=opacity,
                smooth_shading=True,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
            )
            outputs["backbone_actor"] = self.pl.add_mesh(
                outputs["backbone_mesh"],
                **outputs["backbone_kwargs"],
            )

        if show_hsa:
            outputs["hsa_mesh"] = self._generate_rod_mesh(
                T,
                outside_radius=self.rod_params["outside_radius"],
                inside_radius=self.rod_params["outside_radius"] - self.rod_params["wall_thickness"],
            )

            outputs["hsa_kwargs"] = dict(
                color=colors[2],
                opacity=opacity,
                smooth_shading=True,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
            )
            outputs["hsa_actor"] = self.pl.add_mesh(
                outputs["hsa_mesh"], **outputs["hsa_kwargs"]
            )

        if show_orientation_arrows:
            outputs["orientations"] = []
            if orientation_arrow_indices is None:
                outputs["orientation_arrow_indices"] = np.linspace(
                    start=0,
                    stop=T.shape[-1] - 1,
                    num=min(num_orientation_arrows, T.shape[-1]),
                    endpoint=True,
                    dtype=int
                )
            else:
                outputs["orientation_arrow_indices"] = orientation_arrow_indices
            for k in outputs["orientation_arrow_indices"]:
                po_dict = {}
                (
                    po_dict["arrow_meshes"],
                    po_dict["arrow_mesh_kwargs"],
                    po_dict["arrow_actor_kwargs"],
                    po_dict["arrow_actors"],
                ) = self._draw_orientation_arrows(
                    T[..., k],
                    arrow_selector=[True, True, False],
                    opacity=orientation_arrows_opacity if orientation_arrows_opacity is not None else opacity,
                    ambient=ambient,
                    diffuse=diffuse,
                    specular=specular,
                    tip_length=0.225,
                    tip_radius=0.07,
                    shaft_radius=0.04,
                    scale=0.0225,
                )
                outputs["orientations"].append(po_dict)

        return outputs

    def update_meshes(
        self,
        T: np.ndarray,
        outputs: Dict,
    ):
        if "backbone_mesh" in outputs:
            self.pl.remove_actor(outputs["backbone_actor"])
            outputs["backbone_mesh"] = self._generate_rod_mesh(
                T,
                outside_radius=self.backbone_radius,
            )
            outputs["backbone_actor"] = self.pl.add_mesh(
                outputs["backbone_mesh"], **outputs["backbone_kwargs"]
            )

        if "hsa_mesh" in outputs:
            self.pl.remove_actor(outputs["hsa_actor"])
            outputs["hsa_mesh"] = self._generate_rod_mesh(
                T,
                outside_radius=self.rod_params["outside_radius"],
                inside_radius=self.rod_params["outside_radius"] - self.rod_params["wall_thickness"],
            )
            outputs["hsa_actor"] = self.pl.add_mesh(
                outputs["hsa_mesh"], **outputs["hsa_kwargs"]
            )

        if "orientations" in outputs:
            for k, po_dict in zip(outputs["orientation_arrow_indices"], outputs["orientations"]):
                (
                    po_dict["arrow_meshes"],
                    po_dict["arrow_mesh_kwargs"],
                    po_dict["arrow_actor_kwargs"],
                    po_dict["arrow_actors"],
                ) = self._update_orientation_arrows(
                    T[:, :, k],
                    arrow_mesh_kwargs=po_dict["arrow_mesh_kwargs"],
                    arrow_actor_kwargs=po_dict["arrow_actor_kwargs"],
                    arrow_actors=po_dict["arrow_actors"],
                    arrow_selector=[True, True, False],
                )

    def _generate_rod_mesh(
            self,
            T_s: np.ndarray,
            outside_radius: float,
            inside_radius: float = 0.0,
            r_resolution: int = 10,
            phi_resolution: int = 50,
    ) -> pv.StructuredGrid:
        mesh_points = self.rod_mesh_points_fun(
            jnp.array(T_s),
            outside_radius,
            inside_radius,
            r_resolution,
            phi_resolution,
        )

        # reshape T_r from
        #   (4, 4, r_resolution * phi_resolution, z_resolution)
        #   to
        #   (4, 4, r_resolution * phi_resolution * z_resolution)
        mesh_points = mesh_points.reshape((4, 4, -1), order="F")

        grid = pv.StructuredGrid()
        grid.points = np.array(mesh_points[:3, 3, :].T)
        grid.dimensions = [r_resolution, phi_resolution, T_s.shape[-1]]

        return grid

    def _draw_orientation_arrows(
        self,
        T: np.ndarray,
        arrow_mesh_kwargs: List[Dict] = None,
        arrow_actor_kwargs: List[Dict] = None,
        arrow_selector: List[bool] = None,
        opacity: float = 1.0,
        ambient: float = 1.0,
        diffuse: float = 0.5,
        specular: float = 0.0,
        **kwargs,
    ) -> Tuple[List, List[Dict], List[Dict], List]:
        if arrow_selector is None:
            num_arrows = 3
            arrow_selector = [True, True, True]
        else:
            num_arrows = 0
            for val in arrow_selector:
                if val:
                    num_arrows += 1

        if arrow_mesh_kwargs is None:
            arrow_mesh_kwargs = [kwargs for _ in range(num_arrows)]
        if arrow_actor_kwargs is None:
            common_kwargs = dict(
                opacity=opacity,
                smooth_shading=True,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
            )
            arrow_actor_kwargs = [
                common_kwargs | dict(color="red"),
                common_kwargs | dict(color="green"),
                common_kwargs | dict(color="blue"),
            ]

        arrow_meshes = []
        arrow_actors = []
        arrow_idx = 0
        if arrow_selector[0]:
            arrow_meshes.append(
                pv.Arrow(
                    start=T[:3, 3], direction=T[:3, 0], **arrow_mesh_kwargs[arrow_idx]
                )
            )
            arrow_actors.append(
                self.pl.add_mesh(
                    arrow_meshes[arrow_idx], **arrow_actor_kwargs[arrow_idx]
                ),
            )
            arrow_idx += 1
        if arrow_selector[1]:
            arrow_meshes.append(
                pv.Arrow(
                    start=T[:3, 3], direction=T[:3, 1], **arrow_mesh_kwargs[arrow_idx]
                )
            )
            arrow_actors.append(
                self.pl.add_mesh(
                    arrow_meshes[arrow_idx], **arrow_actor_kwargs[arrow_idx]
                ),
            )
            arrow_idx += 1
        if arrow_selector[2]:
            arrow_meshes.append(
                pv.Arrow(
                    start=T[:3, 3], direction=T[:3, 2], **arrow_mesh_kwargs[arrow_idx]
                )
            )
            arrow_actors.append(
                self.pl.add_mesh(
                    arrow_meshes[arrow_idx], **arrow_actor_kwargs[arrow_idx]
                ),
            )
            arrow_idx += 1

        return arrow_meshes, arrow_mesh_kwargs, arrow_actor_kwargs, arrow_actors

    def _update_orientation_arrows(
        self,
        T: np.ndarray,
        arrow_mesh_kwargs: List[Dict],
        arrow_actor_kwargs: List[Dict],
        arrow_actors: List,
        arrow_selector=None,
    ):
        for arrow_actor in arrow_actors:
            self.pl.remove_actor(arrow_actor)

        return self._draw_orientation_arrows(
            T,
            arrow_mesh_kwargs=arrow_mesh_kwargs,
            arrow_actor_kwargs=arrow_actor_kwargs,
            arrow_selector=arrow_selector,
        )

    def close(self):
        self.pl.close()
