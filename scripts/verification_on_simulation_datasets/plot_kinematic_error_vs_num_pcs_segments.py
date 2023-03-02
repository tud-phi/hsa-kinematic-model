import matplotlib

matplotlib.use("Qt5Cairo")

from collections import defaultdict
from jax import numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

datasets_dir = Path("data") / "simulations"
DATASET_NAMES = {
    "elongation": "20221025_170312_elongation_seed-101_100-samples",
    "circles": "20221025_170335_bending_seed-101_225-samples",
    "lemniscate": "20221025_215237_lemniscate_seed-101_100-samples",
    "twisting": "20221025_214951_twisting_seed-101_200-samples",
    "rand. actuation": "20221025_220436_combined_seed-101_500-samples",
}
MAX_NUM_SEGMENTS = 5

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

if __name__ == "__main__":
    results = {}
    for traj_idx, (trajectory_name, dataset_name) in enumerate(DATASET_NAMES.items()):
        dataset_dir = datasets_dir / dataset_name
        dataset_results = defaultdict(list)

        for num_segments in range(1, MAX_NUM_SEGMENTS + 1):
            results_path = (
                dataset_dir / f"inverse_kinematics_results_spcs_n_S-{num_segments}.npz"
            )
            data = jnp.load(str(results_path))

            dataset_results["num_segments"].append(num_segments)
            dataset_results["rmse_quat"].append(data["rmse_quat"])
            dataset_results["rmse_t"].append(data["rmse_t"])
            dataset_results["rmse_euler_xyz"].append(data["rmse_euler_xyz"])

        results[trajectory_name] = dataset_results

    fig, ax = plt.subplots(
        num="Kinematic position error vs. number of PCS segments", figsize=(4.5, 2.5)
    )
    for traj_idx, (trajectory_name, dataset_results) in enumerate(results.items()):
        ax.plot(
            dataset_results["num_segments"],
            jnp.array(dataset_results["rmse_t"]) * 1000,
            color=color_cycle[traj_idx],
            linestyle="solid",
            marker="o",
            label=trajectory_name.capitalize(),
        )
    ax.set_xlabel(r"Number of PCS segments $n_\mathrm{S}$")
    ax.set_ylabel(r"Positional error $e_\mathrm{p}$ [mm]")
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(
        num="Kinematic rotation error vs. number of PCS segments", figsize=(4.5, 2.5)
    )
    for traj_idx, (trajectory_name, dataset_results) in enumerate(results.items()):
        ax.plot(
            dataset_results["num_segments"],
            jnp.array(dataset_results["rmse_quat"]),
            color=color_cycle[traj_idx],
            linestyle="solid",
            marker="o",
            label=trajectory_name.capitalize(),
        )
    ax.set_xlabel(r"Number of PCS segments $n_\mathrm{S}$")
    ax.set_ylabel(r"Rotational error $e_{\mathrm{quat}}$ [-]")
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
