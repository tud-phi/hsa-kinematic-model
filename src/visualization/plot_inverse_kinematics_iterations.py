import jax.numpy as jnp
from jax.numpy.linalg import norm
import matplotlib.pyplot as plt


def plot_inverse_kinematics_iterations(q_its: jnp.ndarray, e_p_its: jnp.ndarray):
    """
    Plot the inverse kinematics iterations.
    :param q_its: array of configuration estimate at each iteration of shape (num_iterations, num_dofs)
    :param e_p_its: array of pose error at each iteration of shape (num_iterations, 7)
    :return:
    """
    fig, axes = plt.subplots(3)
    rmse_quat = jnp.sqrt(jnp.mean(norm(e_p_its[:, :3, :], axis=1) ** 2, axis=1))
    axes[0].plot(q_its[:, 0], label=r"$\kappa_x$")
    axes[0].plot(q_its[:, 1], label=r"$\kappa_y$")
    axes[0].plot(q_its[:, 2], label=r"$\kappa_z$")
    axes[0].set_xlabel("iteration [-]")
    axes[0].set_ylabel("curvature [rad/m]")
    axes[0].legend()
    axes[1].plot(rmse_quat, label=r"$e_{|| \epsilon ||}$")
    axes[1].plot(e_p_its.mean(axis=2)[:, 0], label=r"$e_{\epsilon_x}$")
    axes[1].plot(e_p_its.mean(axis=2)[:, 1], label=r"$e_{\epsilon_y}$")
    axes[1].plot(e_p_its.mean(axis=2)[:, 2], label=r"$e_{\epsilon_z}$")
    axes[1].legend()
    axes[1].set_xlabel("iteration [-]")
    axes[1].set_ylabel("quaternion error [rad]")
    axes[2].plot(e_p_its.mean(axis=2)[:, 4], label=r"$e_{x}$")
    axes[2].plot(e_p_its.mean(axis=2)[:, 5], label=r"$e_{y}$")
    axes[2].plot(e_p_its.mean(axis=2)[:, 6], label=r"$e_{z}$")
    axes[2].legend()
    axes[2].set_xlabel("iteration [-]")
    axes[2].set_ylabel("translation error [m]")
    plt.tight_layout()
    plt.show()
