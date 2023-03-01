import matplotlib

matplotlib.use("Qt5Cairo")
from collections import defaultdict
import spcs_kinematics.jax_math as jmath
from spcs_kinematics.kinematic_parametrizations import SelectivePiecewiseConstantStrain
from jax import jit, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings


from scripts.verification_on_experimental_datasets.plotting_utils import plot_experimental_dataset
from src.visualization import (
    plot_rod_shape,
    plot_inverse_kinematics_iterations,
)


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)


# DATASET_DIR = Path("data/experiments/20221011_174514")  # elongation to 180°
DATASET_DIR = Path("data/experiments/20221011_184131")  # bending to 180°
# DATASET_DIR = Path("data/experiments/20221012_153814")  # lemniscate to 210°
# DATASET_DIR = Path("data/experiments/20221012_103309")  # twisting to 180°
# DATASET_DIR = Path("data/experiments/20221012_140717")  # combined to 180°

CALIBRATE_POSITIONAL_OFFSETS = {
    "r1": True,
    "t23": True,
}
CALIBRATE_ROTATIONAL_OFFSETS = {
    "r1": True,
}
INITIALIZE_CONFIGURATION_AT_LAST_TIMESTEP = True
PLOT_DATASET = True
PLOT_INVERSE_KINEMATICS = True
PLOT_EVERY_NTH_SAMPLE = 20

NUM_PCS_SEGMENTS = 2
STRAIN_SELECTOR_CS = jnp.array([False, False, True, False, False, True])
STRAIN_SELECTOR_PCS = jnp.array([True, True, False, True, True, False])

GAMMA = 2e0 * jnp.ones((1 + STRAIN_SELECTOR_CS.sum() + NUM_PCS_SEGMENTS * STRAIN_SELECTOR_PCS.sum(),))
GAMMA = GAMMA.at[0].set(0.0)
# Increase the step size for the twist strain
# Nominal
GAMMA = GAMMA.at[1].mul(5.0)
# For only PCS
# GAMMA = GAMMA.at[3].mul(5.0).at[9].mul(5.0)
print("GAMMA", GAMMA)

kinematic_settings_postfix = f"_spcs_n_S-{NUM_PCS_SEGMENTS}"

# vmap and jit some functions
vquat_SE3_to_se3 = jit(
    vmap(
        fun=jmath.quat_SE3_to_se3,
        in_axes=0,
        out_axes=0,
    ),
)
vinverse_transformation_matrix = jit(
    vmap(
        fun=jmath.inverse_transformation_matrix,
        in_axes=0,
        out_axes=0,
    )
)
vpose_error_euler = jit(
    vmap(
        jmath.compute_pose_error_euler_xyz_representation,
        in_axes=(2, 2, None),
        out_axes=1,
    )
)
veuler_xyz_to_rotmat = jit(
    vmap(
        jmath.euler_xyz_to_rotmat,
        in_axes=0,
        out_axes=0,
    )
)


# rotation matrix from world frame to base frame
T_WB = jnp.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]
)
T_WB_inv = jmath.inverse_transformation_matrix(T_WB)


@jit
def rotate_y_up_to_z_up(T_W: jnp.ndarray):
    """
    Transform measurements from the  world frame to the base frame.
    :param: T_W: SE(3) transformation matrix in world frame of shape (4, 4)
    :return: T_B SE(3) transformation matrix in base frame of shape (4, 4)
    """
    T_B = T_WB_inv @ T_W @ T_WB
    return T_B


vrotate_y_up_to_z_up = jit(vmap(rotate_y_up_to_z_up, in_axes=0, out_axes=0))


if __name__ == "__main__":
    time_ts = jnp.array(
        pd.read_csv(str(DATASET_DIR / "experiment_time_history.csv")).to_numpy()
    )[:, 0]
    sample_idx_ts = jnp.arange(time_ts.shape[0])

    # Load the motor positions
    motor_positions_ts = jnp.array(
        pd.read_csv(str(DATASET_DIR / "present_positions_history.csv")).to_numpy()
    )
    motor_velocities_ts = jnp.array(
        pd.read_csv(str(DATASET_DIR / "present_velocities_history.csv")).to_numpy()
    )
    motor_loads_ts = jnp.array(
        pd.read_csv(str(DATASET_DIR / "present_loads_history.csv")).to_numpy()
    )
    goal_positions_ts = jnp.array(
        pd.read_csv(str(DATASET_DIR / "goal_positions_history.csv")).to_numpy()
    )

    # Load the MoCap data
    mocap_df = pd.read_csv(str(DATASET_DIR / "mocap_frame_data_history.csv"))
    mocap_df_np = jnp.array(mocap_df.to_numpy())
    mocap_id_ts = mocap_df_np[:, 1]
    # pose of base
    chi_b_ts = mocap_df_np[mocap_id_ts == 2, 2:]
    # pose of platform
    chi_p_ts = mocap_df_np[mocap_id_ts == 3, 2:]
    # pose of marker ring
    chi_r1_ts = mocap_df_np[mocap_id_ts == 4, 2:]

    # correct small differences in the number of samples per data frame
    min_num_samples = min(
        time_ts.shape[0],
        sample_idx_ts.shape[0],
        chi_b_ts.shape[0],
        chi_p_ts.shape[0],
        chi_r1_ts.shape[0],
        motor_positions_ts.shape[0],
        motor_velocities_ts.shape[0],
        motor_loads_ts.shape[0],
        goal_positions_ts.shape[0],
    )
    time_ts = time_ts[:min_num_samples]
    sample_idx_ts = sample_idx_ts[:min_num_samples]
    chi_b_ts = chi_b_ts[:min_num_samples]
    chi_p_ts = chi_p_ts[:min_num_samples]
    chi_r1_ts = chi_r1_ts[:min_num_samples]
    motor_positions_ts = motor_positions_ts[:min_num_samples]
    motor_velocities_ts = motor_velocities_ts[:min_num_samples]
    motor_loads_ts = motor_loads_ts[:min_num_samples]
    goal_positions_ts = goal_positions_ts[:min_num_samples]

    # number of samples
    num_samples = time_ts.shape[0]

    # motor position errors
    motor_position_error_ts = jnp.abs(goal_positions_ts - motor_positions_ts)
    # motor offsets from neutral position
    motor_offsets_ts = motor_positions_ts - motor_positions_ts[0]
    goal_offsets_ts = goal_positions_ts - motor_positions_ts[0]
    # motor angles
    motor_angles_ts = motor_offsets_ts / 2048 * jnp.pi
    motor_angular_velocities_ts = motor_velocities_ts / 2048 * jnp.pi
    goal_angles_ts = goal_offsets_ts / 2048 * jnp.pi
    # rotation angle of rod 23
    phi0_ts = motor_angles_ts[:, 2]

    # poses from SE(3) to se(3) (e.g. transformation matrices)
    T_b_ts = vquat_SE3_to_se3(chi_b_ts)
    T_p_ts = vquat_SE3_to_se3(chi_p_ts)
    T_r1_ts = vquat_SE3_to_se3(chi_r1_ts)

    # switch y-down to z-down
    # rotation matrix from world frame to base frame
    T_b_ts = vrotate_y_up_to_z_up(T_b_ts)
    T_p_ts = vrotate_y_up_to_z_up(T_p_ts)
    T_r1_ts = vrotate_y_up_to_z_up(T_r1_ts)

    # remove the translational offset of the base
    # the MoCap markers of the base are 3.5 mm above the base frame (e.g. top surface of the top motor mounting plate)
    base_marker_z_offset = 3.5e-3  # [m]
    T_b_ts = T_b_ts.at[:, :3, 3].set(
        T_b_ts[:, :3, 3] - jnp.array([0, 0, base_marker_z_offset])
    )
    # transformation matrix from base frame to platform frame
    T_bp_ts = T_p_ts.at[:, :3, 3].set(T_p_ts[:, :3, 3] - T_b_ts[:, :3, 3])
    T_br1_ts = T_r1_ts.at[:, :3, 3].set(T_r1_ts[:, :3, 3] - T_b_ts[:, :3, 3])

    # transformation matrix from base frame of robot to base frame of rod 23
    # we also remove the rigid, proximal end of the rod, which is about 25mm
    T_bb23 = jnp.array(
        [[1, 0, 0, -24e-3], [0, 1, 0, -24e-3], [0, 0, 1, 25e-3], [0, 0, 0, 1]]
    )
    T_b23b = jmath.inverse_transformation_matrix(T_bb23)
    T_b23b_ts = jnp.repeat(jnp.expand_dims(T_b23b, axis=0), num_samples, axis=0)
    # transformation matrix from base frame of rod 23 to 0 frame of rod (e.g. at s=0, which includes the phi0 rotation)
    R_b230_ts = veuler_xyz_to_rotmat(
        jnp.concatenate(
            [jnp.zeros((num_samples, 2)), jnp.expand_dims(phi0_ts, axis=1)], axis=1
        )
    )
    T_b230_ts = jnp.repeat(
        jnp.expand_dims(jnp.identity(4), axis=0), repeats=num_samples, axis=0
    )
    T_b230_ts = T_b230_ts.at[:, :3, :3].set(R_b230_ts)

    # transformation matrix from base frame of rod 23 to marker ring
    T_b23r1_ts = (
        jnp.repeat(jnp.expand_dims(T_b23b, axis=0), num_samples, axis=0) @ T_br1_ts
    )

    # correcting the pose of the marker ring
    T_r1_corr = jnp.identity(4)
    if CALIBRATE_POSITIONAL_OFFSETS.get("r1", False):
        # calibrate the position of the marker ring
        T_r1_corr = T_r1_corr.at[:2, 3].set(T_b23r1_ts[0, :2, 3])
    if CALIBRATE_ROTATIONAL_OFFSETS.get("r1", False):
        # calibrate the rotation of the marker ring
        T_r1_corr = T_r1_corr.at[:3, :3].set(T_b23r1_ts[0, :3, :3])
    T_r1_corr_inv = jmath.inverse_transformation_matrix(T_r1_corr)
    T_b23r1_ts = T_b23r1_ts @ jnp.repeat(
        jnp.expand_dims(T_r1_corr_inv, axis=0), num_samples, axis=0
    )

    # transformation matrix from distal end of rod 23 to marker centroid of platform
    # the rigid, distal end of the rod is about 20mm long
    # the platform have a thickness of 7 mm and the MoCap markers are 7.5 mm above the platform
    # also we need to translate 24mm in x and y direction to the center of the platform
    T_t23p = jnp.array(
        [
            [1, 0, 0, 24e-3],
            [0, 1, 0, 24e-3],
            [0, 0, 1, (20 + 7 + 7.5) * 10 ** (-3)],
            [0, 0, 0, 1],
        ]
    )
    if CALIBRATE_POSITIONAL_OFFSETS.get("t23", False):
        # calibrate x-y offset of re-constructed tip position of rod 23
        T_t23p = T_t23p.at[:2, 3].set(T_t23p[:2, 3] + T_bp_ts[0, :2, 3])

    T_pt23 = jmath.inverse_transformation_matrix(T_t23p)
    T_pt23_ts = jnp.repeat(jnp.expand_dims(T_pt23, axis=0), num_samples, axis=0)
    # transformation matrix from base frame of rod 23 to tip frame of rod 23 using the platform pose information
    T_b23t23_ts = T_b23b_ts @ T_bp_ts @ T_pt23_ts

    # points along the rod in unit [m]
    s = jnp.array([[T_b230_ts[0, 2, 3], T_b23r1_ts[0, 2, 3], T_b23t23_ts[0, 2, 3]]])
    s_ts = jnp.repeat(s, repeats=num_samples, axis=0)

    # combine transformation matrices of points of rod 23
    T_ts = jnp.stack([T_b230_ts, T_b23r1_ts, T_b23t23_ts], axis=-1)

    # collect steady state samples
    indices_sss = []
    delta_goal_offsets_ts = jnp.diff(goal_offsets_ts, axis=0)
    (goal_step_indices,) = jnp.nonzero(jnp.abs(delta_goal_offsets_ts).mean(axis=1) > 0)
    # add first step
    goal_step_indices += 1
    for goal_step_idx in range(goal_step_indices.shape[0]):
        start_idx = goal_step_indices[goal_step_idx]
        end_idx = (
            goal_step_indices[goal_step_idx + 1]
            if goal_step_idx + 1 < goal_step_indices.shape[0]
            else num_samples
        )

        goal_position_cond = (sample_idx_ts >= start_idx) & (sample_idx_ts < end_idx)

        # require steady-state
        steady_state_cond = motor_velocities_ts.mean(axis=1) <= 1
        print(
            f"Checking samples for goal position {goal_offsets_ts[start_idx, :]} "
            f"in time period [{time_ts[start_idx]}, {time_ts[end_idx]}]"
        )

        print(
            f"Checked for time-period and steady-state "
            f"and {(goal_position_cond & steady_state_cond).sum()} samples are valid"
        )
        if (goal_position_cond & steady_state_cond).sum() == 0:
            warnings.warn("No valid samples found for this goal position")
            continue

        # look at which sample idx has the lowest motor position error during time period
        best_subset_sample_idx = jnp.argmin(
            motor_position_error_ts[goal_position_cond & steady_state_cond]
        )
        # look-up global sample idx
        best_sample_idx = sample_idx_ts[goal_position_cond & steady_state_cond][
            best_subset_sample_idx
        ]

        indices_sss.append(best_sample_idx)

    indices_sss = jnp.array(indices_sss)
    print(f"Selected in total {indices_sss.shape[0]} samples from {DATASET_DIR}")

    if PLOT_DATASET:
        plot_experimental_dataset(
            indices_sss,
            time_ts,
            motor_angles_ts,
            goal_angles_ts,
            motor_loads_ts,
            T_b23r1_ts,
            T_b23t23_ts,
            T_bp_ts,
        )

    L0 = T_b23t23_ts[0, 2, 3]
    print("Identified L0 = ", L0 * 1e3, "mm")

    kinematics = SelectivePiecewiseConstantStrain(
        l0=L0 / NUM_PCS_SEGMENTS * jnp.ones((NUM_PCS_SEGMENTS,)),
        strain_selector_cs=STRAIN_SELECTOR_CS,
        strain_selector_pcs=STRAIN_SELECTOR_PCS,
    )

    tmp_output_data = defaultdict(list)
    q_hat = jnp.zeros(GAMMA.shape)
    for sss_idx in range(indices_sss.shape[0]):
        sample_idx = indices_sss[sss_idx]
        print(
            f"Running inverse kinematics for steady state sample idx {sss_idx} / {indices_sss.shape[0]} "
            f"and time {time_ts[sample_idx]}"
        )

        T = T_ts[sample_idx, ...]

        # initialize configuration
        if INITIALIZE_CONFIGURATION_AT_LAST_TIMESTEP:
            q_init = q_hat.copy()
            q_init = q_init.at[0].set(motor_angles_ts[sample_idx, 2])
        else:
            q_init = jnp.zeros(GAMMA.shape)
            q_init = q_init.at[0].set(motor_angles_ts[sample_idx, 2])

        # Run inverse kinematics to estimate the state of the rod
        q_hat, e_chi, q_its, e_chi_its = kinematics.inverse_kinematics(
            T,
            s_ts[sample_idx],
            num_iterations=5000,
            state_init=q_init,
            translational_error_weight=1e0,
            rotational_error_weight=1e0,
            gamma=GAMMA,
        )
        print("Estimated state", q_hat)
        e_quat, e_t = jmath.quat_pose_error_to_rmse(e_chi)

        # use estimated state to compute transformation matrices to points
        plotting_points = jnp.linspace(start=0, stop=s_ts[sample_idx, -1], num=20)
        T_hat = kinematics.forward_kinematics(s_ts[sample_idx], configuration=q_hat)

        # error in euler angles
        e_chi_euler = vpose_error_euler(T, T_hat, kinematics.eps)
        rmse_euler_xyz, _ = jmath.euler_angles_pose_error_to_rmse(e_chi_euler)

        print(f"RMSE errors: e_quat={e_quat}, e_t={e_t}, e_euler_xyz={rmse_euler_xyz}")

        # save data
        tmp_output_data["sample_idx_ss"].append(sample_idx)
        tmp_output_data["time_ss"].append(time_ts[sample_idx])
        tmp_output_data["s_ss"].append(s_ts[sample_idx])
        tmp_output_data["T_ss"].append(T)
        tmp_output_data["T_hat_ss"].append(T_hat)
        tmp_output_data["q_hat_ss"].append(q_hat)
        tmp_output_data["e_chi_ss"].append(e_chi)
        tmp_output_data["rmse_t_ss"].append(e_t)
        tmp_output_data["rmse_quat_ss"].append(e_quat)
        tmp_output_data["rmse_euler_xyz_ss"].append(rmse_euler_xyz)

        if PLOT_INVERSE_KINEMATICS and (
            sss_idx % PLOT_EVERY_NTH_SAMPLE == 0 or sss_idx == indices_sss.shape[0] - 1
        ):
            s_plotting = jnp.linspace(start=0, stop=s_ts[sample_idx, -1], num=20)
            T_hat_plotting = kinematics.forward_kinematics(s_plotting, configuration=q_hat)
            # plot the ground-truth and the estimated rod shape
            plot_rod_shape(T=T, T_hat=T_hat_plotting, oal=0.01)

            plot_inverse_kinematics_iterations(q_its[:, 1:-1], e_chi_its)

    output_data = {}
    for key, val in tmp_output_data.items():
        output_data[key] = jnp.stack(val, axis=0)

    # add kinematic information
    output_data["l0"] = kinematics.l0
    output_data["strain_basis_cs"] = kinematics.strain_basis_cs
    output_data["strain_basis_pcs"] = kinematics.strain_basis_pcs
    output_data["strain_selector_cs"] = kinematics.strain_selector_cs
    output_data["strain_selector_pcs"] = kinematics.strain_selector_pcs
    output_data["rest_strain"] = kinematics.rest_strain

    # compute total error
    output_data["rmse_t"] = jnp.sqrt(jnp.mean(jnp.power(output_data['rmse_t_ss'], 2)))
    output_data["rmse_quat"] = jnp.sqrt(jnp.mean(jnp.power(output_data['rmse_quat_ss'], 2)))
    output_data["rmse_euler_xyz"] = jnp.sqrt(jnp.mean(jnp.power(output_data['rmse_euler_xyz_ss'], 2), axis=0))

    jnp.savez(
        str(DATASET_DIR / f"inverse_kinematics_results{kinematic_settings_postfix}.npz"),
        **output_data,
    )

    print(
        f"Total RMSE of inverse kinematics on dataset: "
        f"e_quat={output_data['rmse_quat']}, "
        f"e_t={output_data['rmse_t']}, "
        f"e_euler_xyz={output_data['rmse_euler_xyz']}"
    )
