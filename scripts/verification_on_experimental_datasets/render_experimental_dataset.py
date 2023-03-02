from jax import jit, vmap
from jax import numpy as jnp
from pathlib import Path
import spcs_kinematics.jax_math as jmath
from spcs_kinematics.kinematic_parametrizations import SelectivePiecewiseConstantStrain

from src.visualization.pyvista_scene import PyvistaScene

# DATASET_DIR = Path("data/experiments/20221011_174514")  # elongation to 180°
# DATASET_DIR = Path("data/experiments/20221011_184131")  # bending to 180°
DATASET_DIR = Path("data/experiments/20221012_153814")  # lemniscate to 210°
# DATASET_DIR = Path("data/experiments/20221012_103309")  # twisting to 180°
# DATASET_DIR = Path("data/experiments/20221012_140717")  # combined to 180°

DATASET_NAME = "lemniscate"

# load data
data = jnp.load(str(DATASET_DIR / "inverse_kinematics_results_spcs_n_S-2.npz"))

l0 = data["l0"]
s_ss = data["s_ss"]
T_ss = data["T_ss"]
T_hat_ss = data["T_hat_ss"]
q_hat_ss = data["q_hat_ss"]
sss_idx = jnp.arange(T_ss.shape[0])

rod_params = dict(
    printed_length=l0.sum(),
    outside_radius=25.4e-3 / 2,  # m
    # wall_thickness=2.43e-3,  # m
    wall_thickness=25.4e-3 / 2,  # m
)


vspcs_forward_kinematics = jit(
    vmap(
        vmap(
            jmath.spcs_forward_kinematics,
            in_axes=(None, None, None, -1, None, None, None),
            out_axes=-1,
        ),
        in_axes=(None, None, None, None, None, 0, None),
        out_axes=0,
    )
)

matmul_vmap_both = jit(vmap(  # vmap for time
    vmap(jnp.matmul, in_axes=(-1, -1), out_axes=-1),  # vmap for points
    in_axes=(0, 0),
    out_axes=0,
))

matmul_vmap_front = jit(vmap(  # vmap for time
    vmap(jnp.matmul, in_axes=(-1, None), out_axes=-1),  # vmap for points
    in_axes=(0, None),
    out_axes=0,
))

matmul_vmap_back = jit(vmap(  # vmap for time
    vmap(jnp.matmul, in_axes=(None, -1), out_axes=-1),  # vmap for points
    in_axes=(None, 0),
    out_axes=0,
))

matmul_vmap_time_back = jit(vmap(  # vmap for time
    vmap(jnp.matmul, in_axes=(None, -1), out_axes=-1),  # vmap for points
    in_axes=(0, 0),
    out_axes=0,
))

if __name__ == "__main__":
    kinematics = SelectivePiecewiseConstantStrain(
        l0=l0,
        strain_selector_cs=data["strain_selector_cs"],
        strain_selector_pcs=data["strain_selector_pcs"],
        rest_strain=data["rest_strain"],
    )

    # clean implementation
    # s_plotting = jnp.linspace(start=0, stop=l0.sum(), num=100)
    # slightly dirty implementation to get ring pose aligned with the orientation arrows we are plotting
    s_plotting_part1 = jnp.linspace(start=s_ss[0, 0], stop=s_ss[0, 1], num=49, endpoint=False)
    s_plotting_part2 = jnp.linspace(start=s_ss[0, 1], stop=s_ss[0, 2], num=51)
    s_plotting = jnp.concatenate([s_plotting_part1, s_plotting_part2])

    T_hat_plotting_ss = vspcs_forward_kinematics(
        kinematics.strain_basis_cs,
        kinematics.strain_basis_pcs,
        kinematics.rest_strain,
        s_plotting,
        l0,
        q_hat_ss,
        kinematics.eps,
    )

    # compute points for rigid proximal end
    s_rigid_proximal_end = jnp.linspace(
        start=0, stop=25e-3, num=10
    ).reshape(1, -1).repeat(T_hat_plotting_ss.shape[0], axis=0)
    T_rigid_proximal_end = T_hat_plotting_ss[..., 0:1].repeat(
        repeats=s_rigid_proximal_end.shape[-1], axis=-1
    )
    T_rigid_proximal_end = T_rigid_proximal_end.at[..., 2, 3, :].add(
        s_rigid_proximal_end
    )

    # offset of 25mm at the proximal end which we assume rigid
    T_rigid_offset = jnp.eye(4)
    T_rigid_offset = T_rigid_offset.at[:3, 3].set(T_rigid_proximal_end[0, :3, 3, -1])
    T_ss = matmul_vmap_back(T_rigid_offset, T_ss)
    T_hat_plotting_ss = matmul_vmap_back(T_rigid_offset, T_hat_plotting_ss)

    # add points of rigid ends
    T_hat_plotting_ss = jnp.concatenate([T_rigid_proximal_end, T_hat_plotting_ss], axis=-1)

    # offset of rod base with respect to global world frame
    T_base_offset = jnp.array([[1, 0, 0, -24e-3], [0, 1, 0, -24e-3], [0, 0, 1, 0.0], [0, 0, 0, 1]])
    T_ss = matmul_vmap_back(T_base_offset, T_ss)
    T_hat_plotting_ss = matmul_vmap_back(T_base_offset, T_hat_plotting_ss)

    # compute points for rigid distal end
    s_rigid_distal_end = jnp.linspace(
        start=0, stop=20e-3, num=10
    ).reshape(1, -1).repeat(T_hat_plotting_ss.shape[0], axis=0)
    T_rigid_distal_end = jnp.eye(4).reshape(1, 4, 4, 1).repeat(
        repeats=s_rigid_distal_end.shape[-1], axis=-1
    ).repeat(
        repeats=s_rigid_distal_end.shape[0], axis=0
    )
    T_rigid_distal_end = T_rigid_distal_end.at[..., 2, 3, :].add(
        s_rigid_distal_end
    )
    T_rigid_distal_end = matmul_vmap_time_back(T_hat_plotting_ss[..., -1], T_rigid_distal_end)
    T_hat_plotting_ss = jnp.concatenate([T_hat_plotting_ss, T_rigid_distal_end], axis=-1)

    # indices of orientation arrows for kinematics
    orientation_arrow_indices = s_rigid_proximal_end.shape[-1] + jnp.linspace(
        start=0,
        stop=s_plotting.shape[-1],
        num=11,
        endpoint=True,
        dtype=int
    )

    pv_scene = PyvistaScene(
        rod_params,
        gt_settings=dict(
            show_hsa=False,
            show_orientation_arrows=True,
            num_orientation_arrows=T_ss.shape[-1],
            opacity=1.0,
            orientation_arrows_opacity=1.0,
        ),
        hat_settings=dict(
            show_hsa=True,
            show_orientation_arrows=True,
            orientation_arrow_indices=orientation_arrow_indices,
            opacity=1.0,
            orientation_arrows_opacity=0.2,
            diffuse=1.0,
            ambient=1.0,
            specular=0.8,
        ),
        tip_down=True,
        floor_center=(0, 0, 0.0),  # m
        floor_size=(0.0955, 0.137),  # m
        light_position=(0, 0.2, 0.4),  # m
    )
    # pv_scene.run(T_gt=T_ss[4], T_hat=T_hat_plotting_ss[4])
    pv_scene.animate(
        T_gt_ts=T_ss,
        T_hat_ts=T_hat_plotting_ss,
        filepath=f"scripts/verification_on_experimental_datasets/videos/{DATASET_NAME}.mp4",
        sample_rate=10,
        frame_rate=10,
    )
