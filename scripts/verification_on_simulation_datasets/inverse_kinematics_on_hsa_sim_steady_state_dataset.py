"""
Run inverse kinematics on a dataset for the steady-states of the simulated HSA robot.
"""
from collections import defaultdict
from spcs_kinematics.kinematic_parametrizations import SelectivePiecewiseConstantStrain
import spcs_kinematics.jax_math as jmath
from hsa_sim.utils.data_mapping_utils import (
    simulation_diagnostic_arrays_to_transformation_matrices,
)
import jax
from jax import numpy as jnp
from natsort import natsorted
import pathlib
from typing import *
import yaml

from src.visualization import (
    plot_rod_shape,
    plot_inverse_kinematics_iterations,
)

vpose_error_euler = jax.jit(
    jax.vmap(
        jmath.compute_pose_error_euler_xyz_representation,
        in_axes=(2, 2, None),
        out_axes=1,
    )
)


dataset_name = "20221025_170312_elongation_seed-101_100-samples"
# dataset_name = "20221025_170335_bending_seed-101_225-samples"
# dataset_name = "20221025_215237_lemniscate_seed-101_100-samples"
# dataset_name = "20221025_214951_twisting_seed-101_200-samples"
# dataset_name = "20221025_220436_combined_seed-101_500-samples"

dataset_dir = pathlib.Path(
    f"../hsa_sim/data/hsa_robot/kinematic_steady_state_datasets/{dataset_name}"
)

# either int to select rod_idx or None to select all rods
ROD_INDICES = jnp.arange(start=0, stop=4)

# inverse kinematics parameters
INITIALIZE_CONFIGURATION_AT_LAST_SAMPLE = True
NUM_PCS_SEGMENTS = 2
STRAIN_SELECTOR_CS = jnp.array([False, False, True, False, False, True])
STRAIN_SELECTOR_PCS = jnp.array([True, True, False, True, True, False])

GAMMA = 2e-1 * jnp.ones((1 + STRAIN_SELECTOR_CS.sum() + NUM_PCS_SEGMENTS * STRAIN_SELECTOR_PCS.sum(),))
# Nominal
GAMMA = GAMMA.at[1].mul(5.0)
# For only PCS
# GAMMA = GAMMA.at[3].mul(5.0).at[9].mul(5.0)

kinematic_settings_postfix = f"_spcs_n_S-{NUM_PCS_SEGMENTS}"

PLOT_INVERSE_KINEMATICS = True
PLOT_EVERY_NTH_SAMPLE = 25

if __name__ == "__main__":
    subfolders_list = natsorted(
        [f for f in list(dataset_dir.glob("*")) if f.is_dir()], key=str
    )
    num_samples = len(subfolders_list)
    print(f"Found {num_samples} samples in {dataset_dir}")

    kinematics: Optional[SelectivePiecewiseConstantStrain] = None

    tmp_output_data = defaultdict(list)
    q_hat_prior = [jnp.zeros(GAMMA.shape) for _ in ROD_INDICES]
    for sample_idx, sample_dir in enumerate(subfolders_list):
        with open(f"{str(sample_dir)}/robot_params.yaml", "r") as file:
            robot_params = yaml.safe_load(file)
        # printed length
        L0 = robot_params["segments"][0]["printed_length"]

        rod_diagnostic_arrays = jnp.load(f"{str(sample_dir)}/rod_diagnostic_arrays.npz")
        # transform diagnostic arrays to transformation matrices
        T_ts, _ = simulation_diagnostic_arrays_to_transformation_matrices(
            rod_diagnostic_arrays=rod_diagnostic_arrays
        )
        # construct jnp array
        T_ts = jnp.array(T_ts)
        # remove xy translation of base
        T_ts = T_ts.at[..., 0:2, 3, :].add(
            -jnp.repeat(T_ts[..., 0:2, 3, 0:1], repeats=T_ts.shape[-1], axis=-1)
        )

        if kinematics is None:
            kinematics = SelectivePiecewiseConstantStrain(
                l0=L0 / NUM_PCS_SEGMENTS * jnp.ones((NUM_PCS_SEGMENTS,)),
                strain_selector_cs=STRAIN_SELECTOR_CS,
                strain_selector_pcs=STRAIN_SELECTOR_PCS,
            )
        else:
            assert jnp.allclose(L0, kinematics.l0.sum()), \
            f"Rod printed length ({L0}) changed with respect to initialized kinematics ({kinematics.l0.sum()})!"

        for rod_idx, j in enumerate(ROD_INDICES):
            print(
                f"Running inverse kinematics for sample_idx {sample_idx} / {len(subfolders_list)} and j={j}"
            )

            # extract the last time-step of the rod with index rod_idx
            T = T_ts[-1, 0, j, ...]
            num_nodes = T.shape[-1] + 1

            # initialize configuration
            if INITIALIZE_CONFIGURATION_AT_LAST_SAMPLE:
                q_init = q_hat_prior[rod_idx]
            else:
                q_init = jnp.zeros(GAMMA.shape)

            s = jnp.linspace(start=0.0, stop=L0, num=num_nodes)
            # transform node points to link center points
            s = 0.5 * s[:-1] + 0.5 * s[1:]

            # Run inverse kinematics to estimate the state of the rod
            q_hat, e_chi, q_its, e_chi_its = kinematics.inverse_kinematics(
                T,
                s,
                num_iterations=5000,
                state_init=q_init,
                translational_error_weight=1e0,
                rotational_error_weight=1e0,
                gamma=GAMMA,
            )
            print("Estimated state", q_hat)
            e_quat, e_t = jmath.quat_pose_error_to_rmse(e_chi)

            # save q_hat as q_hat_prior
            q_hat_prior[rod_idx] = q_hat

            # use estimated state to compute transformation matrices to points
            T_hat = kinematics.forward_kinematics(s, configuration=q_hat)

            # error in euler angles
            e_chi_euler = vpose_error_euler(T, T_hat, kinematics.eps)
            rmse_euler_xyz, _ = jmath.euler_angles_pose_error_to_rmse(e_chi_euler)

            print(
                f"RMSE errors: e_quat={e_quat}, e_t={e_t}, e_euler_xyz={rmse_euler_xyz}"
            )

            tmp_output_data["sample_idx_ss"].append(sample_idx)
            tmp_output_data["rod_idx_ss"].append(j)
            tmp_output_data["s_ss"].append(s)
            tmp_output_data["T_ss"].append(T)
            tmp_output_data["T_hat_ss"].append(T_hat)
            tmp_output_data["q_hat_ss"].append(q_hat)
            tmp_output_data["e_chi_ss"].append(e_chi)
            tmp_output_data["rmse_t_ss"].append(e_t)
            tmp_output_data["rmse_quat_ss"].append(e_quat)
            tmp_output_data["rmse_euler_xyz_ss"].append(rmse_euler_xyz)

            if PLOT_INVERSE_KINEMATICS and (
                sample_idx % PLOT_EVERY_NTH_SAMPLE == 0 or sample_idx == len(subfolders_list) - 1
            ):
                # plot the ground-truth and the estimated rod shape
                plot_rod_shape(T=T, T_hat=T_hat, oal=0.01)

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

    inverse_kinematics_results_path = str(dataset_dir / f"inverse_kinematics_results{kinematic_settings_postfix}.npz")
    jnp.savez(
        inverse_kinematics_results_path,
        **output_data,
    )

    print("Saved kinematic results to", inverse_kinematics_results_path)

    print(
        f"Total RMSE of inverse kinematics on dataset: "
        f"e_quat={output_data['rmse_quat']}, "
        f"e_t={output_data['rmse_t']}, "
        f"e_euler_xyz={output_data['rmse_euler_xyz']}"
    )
