from jax import numpy as jnp
import matplotlib
from scipy.spatial.transform import Rotation

matplotlib.use("Qt5Cairo")
import matplotlib.pyplot as plt


def plot_experimental_dataset(
    indices_sss: jnp.ndarray,
    time_ts: jnp.ndarray,
    motor_angles_ts: jnp.ndarray,
    goal_angles_ts: jnp.ndarray,
    motor_loads_ts: jnp.ndarray,
    T_b23r1_ts: jnp.ndarray,
    T_b23t23_ts: jnp.ndarray,
    T_bp_ts: jnp.ndarray,
    motor_max_torque: float = 2.5,
):
    plt.figure(num="Motor angles")
    plt.plot(time_ts, motor_angles_ts[:, 0] / jnp.pi * 180, label=r"$u_{21}$")
    plt.plot(time_ts, motor_angles_ts[:, 1] / jnp.pi * 180, label=r"$u_{22}$")
    plt.plot(time_ts, motor_angles_ts[:, 2] / jnp.pi * 180, label=r"$u_{23}$")
    plt.plot(time_ts, motor_angles_ts[:, 3] / jnp.pi * 180, label=r"$u_{24}$")
    plt.gca().set_prop_cycle(None)
    plt.plot(
        time_ts,
        goal_angles_ts[:, 0] / jnp.pi * 180,
        linestyle=":",
        label=r"$u_{\mathrm{d},21}$",
    )
    plt.plot(
        time_ts,
        goal_angles_ts[:, 1] / jnp.pi * 180,
        linestyle=":",
        label=r"$u_{\mathrm{d},22}$",
    )
    plt.plot(
        time_ts,
        goal_angles_ts[:, 2] / jnp.pi * 180,
        linestyle=":",
        label=r"$u_{\mathrm{d},23}$",
    )
    plt.plot(
        time_ts,
        goal_angles_ts[:, 3] / jnp.pi * 180,
        linestyle=":",
        label=r"$u_{\mathrm{d},24}$",
    )
    plt.vlines(
        x=time_ts[indices_sss],
        ymin=motor_angles_ts.min() / jnp.pi * 180,
        ymax=motor_angles_ts.max() / jnp.pi * 180,
        color="black",
        linestyle="--",
        linewidth=0.5,
        label="selected samples",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Motor angle [deg]")
    plt.legend()
    plt.show()

    plt.figure(num="Motor loads")
    plt.plot(
        time_ts,
        motor_loads_ts[:, 0] / 1000 * motor_max_torque,
        linestyle="-",
        label=r"$\tau_{21}$",
    )
    plt.plot(
        time_ts,
        motor_loads_ts[:, 1] / 1000 * motor_max_torque,
        linestyle="-",
        label=r"$\tau_{22}$",
    )
    plt.plot(
        time_ts,
        motor_loads_ts[:, 2] / 1000 * motor_max_torque,
        linestyle="-",
        label=r"$\tau_{23}$",
    )
    plt.plot(
        time_ts,
        motor_loads_ts[:, 3] / 1000 * motor_max_torque,
        linestyle="-",
        label=r"$\tau_{24}$",
    )
    plt.vlines(
        x=time_ts[indices_sss],
        ymin=motor_loads_ts.min() / 1000 * motor_max_torque,
        ymax=motor_loads_ts.max() / 1000 * motor_max_torque,
        color="black",
        linestyle="--",
        linewidth=0.5,
        label="selected samples",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Motor loads [Nm]")
    plt.legend()
    plt.show()

    # plt.figure(num="Motor 23 velocities")
    # plt.plot(time_ts, motor_angular_velocities_ts[:, 2] / jnp.pi * 180, label=r"$\dot{u}_{23}$")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Motor velocity [deg / s]")
    # plt.legend()
    # plt.show()

    fig, ax1 = plt.subplots(num="Movement of platform")
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax1.plot(
        time_ts,
        T_bp_ts[:, 0, 3] * 1e3,
        color=color_cycle[0],
        label=r"$x_{\mathrm{p}}$",
    )
    ax1.plot(
        time_ts,
        T_bp_ts[:, 1, 3] * 1e3,
        color=color_cycle[1],
        label=r"$y_{\mathrm{p}}$",
    )
    ax1.plot(
        time_ts,
        T_bp_ts[:, 2, 3] * 1e3,
        color=color_cycle[2],
        label=r"$z_{\mathrm{p}}$",
    )
    ax1.grid()
    ax1.set_xlabel(r"Time [s]")
    ax1.set_ylabel(r"Position [mm]")
    ax2 = ax1.twinx()
    xyz_euler_angles = xyz_euler_angles = Rotation.from_matrix(
        T_bp_ts[:, :3, :3]
    ).as_euler("xyz", degrees=False)
    ax2.plot(
        time_ts,
        xyz_euler_angles[:, 0] / jnp.pi * 180,
        color=color_cycle[3],
        label=r"$\alpha_{\mathrm{p}}$",
    )
    ax2.plot(
        time_ts,
        xyz_euler_angles[:, 1] / jnp.pi * 180,
        color=color_cycle[4],
        label=r"$\beta_{\mathrm{p}}$",
    )
    ax2.plot(
        time_ts,
        xyz_euler_angles[:, 2] / jnp.pi * 180,
        color=color_cycle[5],
        label=r"$\gamma_{\mathrm{p}}$",
    )
    ax2.set_ylabel(r"Euler angles XYZ [deg]")
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    plt.legend(ax1_handles + ax2_handles, ax1_labels + ax2_labels)
    plt.show()

    plt.figure(num="Positions of rod 23")
    plt.plot(
        time_ts,
        T_b23r1_ts[:, 0, 3] * 1e3,
        linestyle=":",
        label=r"$x_{\mathrm{r}1}$",
    )
    plt.plot(
        time_ts,
        T_b23r1_ts[:, 1, 3] * 1e3,
        linestyle=":",
        label=r"$y_{\mathrm{r}1}$",
    )
    plt.plot(
        time_ts,
        T_b23r1_ts[:, 2, 3] * 1e3,
        linestyle=":",
        label=r"$z_{\mathrm{r}1}$",
    )
    plt.plot(
        time_ts,
        T_b23t23_ts[:, 0, 3] * 1e3,
        linestyle="--",
        label=r"$x_{\mathrm{t}}$",
    )
    plt.plot(
        time_ts,
        T_b23t23_ts[:, 1, 3] * 1e3,
        linestyle="--",
        label=r"$y_{\mathrm{t}}$",
    )
    plt.plot(
        time_ts,
        T_b23t23_ts[:, 2, 3] * 1e3,
        linestyle="--",
        label=r"$z_{\mathrm{t}}$",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Position [mm]")
    plt.legend()
    plt.show()
