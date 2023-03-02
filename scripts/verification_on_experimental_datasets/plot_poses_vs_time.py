import matplotlib

matplotlib.use("Qt5Cairo")

from jax import jit, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import spcs_kinematics.jax_math as jmath

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

# DATASET_DIR = Path("data/experiments/20221011_174514")  # elongation to 180°
# DATASET_DIR = Path("data/experiments/20221011_184131")  # bending to 180°
DATASET_DIR = Path("data/experiments/20221012_153814")  # lemniscate to 210°
# DATASET_DIR = Path("data/experiments/20221012_103309")  # twisting to 180°
# DATASET_DIR = Path("data/experiments/20221012_140717")  # combined to 180°

# load data
data = jnp.load(str(DATASET_DIR / "inverse_kinematics_results_spcs_n_S-2.npz"))

# number of samples which are cut-off at the end of the dataset
num_cut_off_samples = 1
T_ss = data["T_ss"][:-num_cut_off_samples]
T_hat_ss = data["T_hat_ss"][:-num_cut_off_samples]
sss_idx = jnp.arange(T_ss.shape[0])


def plot_position_vs_time():
    plt.figure(figsize=(4.5, 3))
    ax = plt.gca()

    linewidth_hat = 3
    # plt.plot(
    #     sss_idx,
    #     T_hat_ss[:, 0, 3, 0] * 1000,
    #     linewidth=linewidth_hat,
    #     linestyle=":",
    #     label=r"$\hat{x}_{0}$",
    # )
    # plt.plot(
    #     sss_idx,
    #     T_hat_ss[:, 1, 3, 0] * 1000,
    #     linewidth=linewidth_hat,
    #     linestyle=":",
    #     label=r"$\hat{y}_{0}$",
    # )
    # plt.plot(
    #     sss_idx,
    #     T_hat_ss[:, 2, 3, 0] * 1000,
    #     linewidth=linewidth_hat,
    #     linestyle=":",
    #     label=r"$\hat{z}_{0}$",
    # )
    plt.plot(
        sss_idx,
        T_hat_ss[:, 0, 3, 1] * 1000,
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{x}_{\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        T_hat_ss[:, 1, 3, 1] * 1000,
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{y}_{\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        T_hat_ss[:, 2, 3, 1] * 1000,
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{z}_{\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        T_hat_ss[:, 0, 3, 2] * 1000,
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{x}_{\mathrm{d}}$",
    )
    plt.plot(
        sss_idx,
        T_hat_ss[:, 1, 3, 2] * 1000,
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{y}_{\mathrm{d}}$",
    )
    plt.plot(
        sss_idx,
        T_hat_ss[:, 2, 3, 2] * 1000,
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{z}_{\mathrm{d}}$",
    )

    ax.set_prop_cycle(None)

    linewidth_gt = 1.5
    # plt.plot(
    #     sss_idx, T_ss[:, 0, 3, 0] * 1000,
    #     linewidth=linewidth_gt,
    #     label=r"$x_{0}$"
    # )
    # plt.plot(
    #     sss_idx,
    #     T_ss[:, 1, 3, 0] * 1000,
    #     linewidth=linewidth_gt,
    #     label=r"$y_{0}$"
    # )
    # plt.plot(
    #     sss_idx, T_ss[:, 2, 3, 0] * 1000,
    #     linewidth=linewidth_gt,
    #     label=r"$z_{0}$"
    # )
    plt.plot(
        sss_idx,
        T_ss[:, 0, 3, 1] * 1000,
        linewidth=linewidth_gt,
        label=r"$x_{\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        T_ss[:, 1, 3, 1] * 1000,
        linewidth=linewidth_gt,
        label=r"$y_{\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        T_ss[:, 2, 3, 1] * 1000,
        linewidth=linewidth_gt,
        label=r"$z_{\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        T_ss[:, 0, 3, 2] * 1000,
        linewidth=linewidth_gt,
        label=r"$x_{\mathrm{d}}$",
    )
    plt.plot(
        sss_idx,
        T_ss[:, 1, 3, 2] * 1000,
        linewidth=linewidth_gt,
        label=r"$y_{\mathrm{d}}$",
    )
    plt.plot(
        sss_idx,
        T_ss[:, 2, 3, 2] * 1000,
        linewidth=linewidth_gt,
        label=r"$z_{\mathrm{d}}$",
    )


    plt.xlabel(r"Sample index $t$ [-]")
    plt.ylabel(r"Position in $\{ S_{\mathcal{B}}\}$ frame [mm]")
    plt.grid(True)

    # plt.legend(ncol=2, loc="upper right")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles[int(len(labels) // 2):],
        labels[int(len(labels) // 2):],
        loc="upper right",
        ncol=2,
    )

    plt.tight_layout()
    plt.show()


def plot_position_error_vs_time():
    plt.figure(figsize=(4.5, 3))
    ax = plt.gca()

    # plt.plot(
    #     sss_idx, (T_ss[:, 0, 3, 0] - T_hat_ss[:, 0, 3, 0]) * 1000, label=r"$e_{x,0}$"
    # )
    # plt.plot(
    #     sss_idx, (T_ss[:, 1, 3, 0] - T_hat_ss[:, 1, 3, 0]) * 1000, label=r"$e_{y,0}$"
    # )
    # plt.plot(
    #     sss_idx, (T_ss[:, 2, 3, 0] - T_hat_ss[:, 2, 3, 0]) * 1000, label=r"$e_{z,0}$"
    # )
    plt.plot(
        sss_idx,
        (T_ss[:, 0, 3, 1] - T_hat_ss[:, 0, 3, 1]) * 1000,
        label=r"$e_{x,\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        (T_ss[:, 1, 3, 1] - T_hat_ss[:, 1, 3, 1]) * 1000,
        label=r"$e_{y,\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        (T_ss[:, 2, 3, 1] - T_hat_ss[:, 2, 3, 1]) * 1000,
        label=r"$e_{z,\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        (T_ss[:, 0, 3, 2] - T_hat_ss[:, 0, 3, 2]) * 1000,
        label=r"$e_{x,\mathrm{d}}$",
    )
    plt.plot(
        sss_idx,
        (T_ss[:, 1, 3, 2] - T_hat_ss[:, 1, 3, 2]) * 1000,
        label=r"$e_{y,\mathrm{d}}$",
    )
    plt.plot(
        sss_idx,
        (T_ss[:, 2, 3, 2] - T_hat_ss[:, 2, 3, 2]) * 1000,
        label=r"$e_{z,\mathrm{d}}$",
    )

    plt.xlabel(r"Sample index $t$ [-]")
    plt.ylabel(r"Position error [mm]")
    plt.grid(True)

    plt.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    plt.show()


def plot_euler_xyz_vs_time():
    plt.figure(figsize=(4.5, 3))
    ax = plt.gca()

    vrotmat_to_euler_xyz = jit(
        vmap(
            vmap(
                jmath.rotmat_to_euler_xyz,
                in_axes=0,
                out_axes=0,
            ),
            in_axes=-1,
            out_axes=-1,
        )
    )

    euler_xyz_ss = vrotmat_to_euler_xyz(T_ss[:, :3, :3, :])
    euler_xyz_hat_ss = vrotmat_to_euler_xyz(T_hat_ss[:, :3, :3, :])

    linewidth_hat = 3
    # plt.plot(sss_idx, euler_xyz_hat_ss[:, 0, 0], linewidth=linewidth_hat, linestyle=":", label=r"$\hat{x}_{0}$")
    # plt.plot(sss_idx, euler_xyz_hat_ss[:, 1, 0], linewidth=linewidth_hat, linestyle=":", label=r"$\hat{y}_{0}$")
    # plt.plot(sss_idx, euler_xyz_hat_ss[:, 2, 0], linewidth=linewidth_hat, linestyle=":", label=r"$\hat{z}_{0}$")
    plt.plot(
        sss_idx,
        euler_xyz_hat_ss[:, 0, 1],
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{x}_{\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        euler_xyz_hat_ss[:, 1, 1],
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{y}_{\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        euler_xyz_hat_ss[:, 2, 1],
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{z}_{\mathrm{r}}$",
    )
    plt.plot(
        sss_idx,
        euler_xyz_hat_ss[:, 0, 2],
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{x}_{\mathrm{d}}$",
    )
    plt.plot(
        sss_idx,
        euler_xyz_hat_ss[:, 1, 2],
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{y}_{\mathrm{d}}$",
    )
    plt.plot(
        sss_idx,
        euler_xyz_hat_ss[:, 2, 2],
        linewidth=linewidth_hat,
        linestyle=":",
        label=r"$\hat{z}_{\mathrm{d}}$",
    )

    ax.set_prop_cycle(None)

    linewidth_gt = 1.5
    # plt.plot(sss_idx, euler_xyz_ss[:, 0, 0], linewidth=linewidth_gt, label=r"$x_{0}$")
    # plt.plot(sss_idx, euler_xyz_ss[:, 1, 0], linewidth=linewidth_gt, label=r"$y_{0}$")
    # plt.plot(sss_idx, euler_xyz_ss[:, 2, 0], linewidth=linewidth_gt, label=r"$z_{0}$")
    plt.plot(sss_idx, euler_xyz_ss[:, 0, 1], linewidth=linewidth_gt, label=r"$\alpha_{\mathrm{r}}$")
    plt.plot(sss_idx, euler_xyz_ss[:, 1, 1], linewidth=linewidth_gt, label=r"$\beta_{\mathrm{r}}$")
    plt.plot(sss_idx, euler_xyz_ss[:, 2, 1], linewidth=linewidth_gt, label=r"$\gamma_{\mathrm{r}}$")
    plt.plot(sss_idx, euler_xyz_ss[:, 0, 2], linewidth=linewidth_gt, label=r"$\alpha_{\mathrm{t}}$")
    plt.plot(sss_idx, euler_xyz_ss[:, 1, 2], linewidth=linewidth_gt, label=r"$\beta_{\mathrm{t}}$")
    plt.plot(sss_idx, euler_xyz_ss[:, 2, 2], linewidth=linewidth_gt, label=r"$\gamma_{\mathrm{t}}$")


    plt.xlabel(r"Sample index $t$ [-]")
    plt.ylabel(r"Euler XYZ angles in $\{ S_{\mathcal{B}}\}$ frame [rad]")
    plt.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles[int(len(labels) // 2):],
        labels[int(len(labels) // 2):],
        loc="upper right",
        ncol=2,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_position_vs_time()
    plot_position_error_vs_time()
    plot_euler_xyz_vs_time()
