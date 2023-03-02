from functools import partial
from spcs_kinematics.kinematic_parametrizations import SelectivePiecewiseConstantStrain
from jax import numpy as jnp
import timeit

# number of segments
NUM_SEGMENTS = 2

# length of each segment
l0 = jnp.ones((NUM_SEGMENTS,))

# specify the number of points along the rod
s = jnp.linspace(0, l0.sum(), 3)

# number of inverse kinematics iterations
NUM_ITERATIONS = 1000

# number of timeit runs
NUM_RUNS = 100

if __name__ == "__main__":
    kinematics = SelectivePiecewiseConstantStrain(
        # specify the original length of each segment
        # the rod is composed of 2 segments of length 0.5 m each
        l0=l0,
        # model twist and elongation to be constant across entire rod
        strain_selector_cs=jnp.array([False, False, True, False, False, True]),
        # model the bending and shear strains to be constant across each segment (i.e. piecewise constant)
        strain_selector_pcs=jnp.array([True, True, False, True, True, False]),
    )

    # specify the ground-truth configuration of the rod
    q = jnp.zeros(kinematics.configuration.shape)

    # run forward kinematics
    T = kinematics.forward_kinematics(s, configuration=q)

    # initial guess for the configuration vector
    q_init = jnp.zeros_like(q)

    # step size of gradient descent
    gamma = 2e0 * jnp.ones(
        (1 + kinematics.strain_selector_cs.sum() + NUM_SEGMENTS * kinematics.strain_selector_pcs.sum(),)
    )

    # kwargs for inverse kinematics
    inverse_kinematics_fn = partial(
        kinematics.inverse_kinematics,
        transformations=T,
        points=s,
        num_iterations=NUM_ITERATIONS,
        state_init=q_init,
        translational_error_weight=1e0,
        rotational_error_weight=1e0,
        gamma=gamma,
    )

    # run inverse kinematics once to jit-compile the function
    q_hat, e_chi, q_its, e_chi_its = inverse_kinematics_fn()

    print(f"First run of inverse kinematics is done. Now running {NUM_RUNS} times to benchmark performance...")

    # evaluate the duration of running inverse kinematics
    duration = timeit.timeit(inverse_kinematics_fn, number=NUM_RUNS)

    # compute the average duration per iteration of inverse kinematics
    avg_dur_per_it = duration / (NUM_RUNS * NUM_ITERATIONS)

    print("Average duration of running inverse kinematics "
          "for one iteration: {:.0f} microseconds".format(avg_dur_per_it * 1e6))
