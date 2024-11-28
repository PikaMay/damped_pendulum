import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax
import os
import hydra
from omegaconf import DictConfig


def ode_func(y, t, b, m, l, g):
    """Defines the ODE dynamics for the pendulum.

    Args:
      y: State vector [theta, dtheta/dt].
      t: Time (not explicitly used in this ODE, but required for compatibility).
      b: Damping coefficient.
      m: Mass of the pendulum.
      l: Length of the pendulum.
      g: Acceleration due to gravity.

    Returns:
      Derivative of the state vector [dtheta/dt, d2theta/dt2].
    """
    theta, omega = y  # unpacking the state vector
    dtheta_dt = omega  # Angular velocity
    domega_dt = -(b / (m * l)) * omega - (g / l) * jnp.sin(
        theta
    )  # Angular acceleration

    return jnp.array([dtheta_dt, domega_dt])


#    pass  # TODO: Complete this function


def solve_pendulum_euler(y0, t_span, dt, b, m, l, g):
    """Solves the pendulum equation using the Euler method.

    Args:
      y0: Initial conditions [theta, dtheta/dt] at t=0.
      t_span: Tuple (t0, t_end) representing the time interval.
      dt: Time step.
      b: Damping coefficient.
      m: Mass of the pendulum.
      l: Length of the pendulum.
      g: Acceleration due to gravity.

    Returns:
      A tuple (t, y) where:
        t: Array of time points.
        y: Array of solution values at each time point.
    """

    t0, t_end = t_span
    t = jnp.arange(t0, t_end, dt)

    # Initial state
    y_init = y0

    def scan_fn(carry, t):
        """scan function to perfom Euler"""
        y, t_curr = carry
        y_next = y + dt * ode_func(y, t_curr, b, m, l, g)
        return (y_next, t_curr + dt), y_next

    carry_init = (y_init, t[0])
    _, output = lax.scan(scan_fn, carry_init, t)

    return t, output


def solve_pendulum_rk(y0, t_span, dt, b, m, l, g):
    """Solves the pendulum equation using the Runge Kutta method.

    Args:
      y0: Initial conditions [theta, dtheta/dt] at t=0.
      t_span: Tuple (t0, t_end) representing the time interval.
      dt: Time step.
      b: Damping coefficient.
      m: Mass of the pendulum.
      l: Length of the pendulum.
      g: Acceleration due to gravity.

    Returns:
      A tuple (t, y) where:
        t: Array of time points.
        y: Array of solution values at each time point.
    """
    t0, t_end = t_span  # Unpacking the time interval from start time to end time
    t = jnp.arange(t0, t_end, dt)  # Time array, evenly spaced time points, step size dt
    y_init = y0

    def rk4_step(y, t, dt, b, m, l, g):
        """Performs a single step of the RK4 method"""
        k1 = dt * ode_func(y, t, b, m, l, g)
        k2 = dt * ode_func(y + 0.5 * k1, t + 0.5 * dt, b, m, l, g)
        k3 = dt * ode_func(y + 0.5 * k2, t + 0.5 * dt, b, m, l, g)
        k4 = dt * ode_func(y + k3, t + dt, b, m, l, g)

        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def scan_fn(carry, t):
        """Scan function to perform RK4 integration"""
        y, t_curr = carry  # Unpack the carry
        y_next = rk4_step(y, t_curr, dt, b, m, l, g)
        return (y_next, t_curr + dt), y_next

    # Initial carry state
    carry_init = (y_init, t[0])

    _, output = lax.scan(scan_fn, carry_init, t)

    return t, output


#    pass  # TODO: Complete this function


def gen_data(t, y):
    """Generate test and train data from the solution of the numerical method."""
    t_sliced, y_sliced = (
        t[jnp.arange(t.size, step=200)],
        y[jnp.arange(t.size, step=200)],
    )
    split_index = int(0.8 * len(t_sliced))
    t_train, y_train = t_sliced[:split_index], y_sliced[:split_index, 0]
    t_test, y_test = t_sliced[split_index:], y_sliced[split_index:, 0]
    return t_train, y_train, t_test, y_test


def plot_pendulum_results(
    t_euler,
    y_euler,
    t_rk,
    y_rk,
    save_dir="./pendulum_plots",
    plot_filename="pendulum_results.png",
):
    """
    Function to plot and save the pendulum simulation results.

    Parameters:
    - t_euler (array): Time array for the Euler method.
    - y_euler (array): Solution array [theta, omega] for the Euler method.
    - t_rk (array): Time array for the RK4 method.
    - y_rk (array): Solution array [theta, omega] for the RK4 method.
    - save_dir (str): Directory to save the plot.
    - plot_filename (str): The filename to save the plot.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create the full path to save the plot
    plot_file_path = os.path.join(save_dir, plot_filename)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot Angular Position (Theta)
    plt.subplot(2, 1, 1)
    plt.plot(t_euler, y_euler[:, 0], label="Euler Method (Theta)", color="b")
    plt.plot(t_rk, y_rk[:, 0], label="RK4 Method (Theta)", color="r", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.title("Pendulum Angular Position (Theta)")
    plt.legend()
    plt.grid()

    # Plot Angular Velocity (Omega)
    plt.subplot(2, 1, 2)
    plt.plot(t_euler, y_euler[:, 1], label="Euler Method (Omega)", color="b")
    plt.plot(t_rk, y_rk[:, 1], label="RK4 Method (Omega)", color="r", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Omega (rad/s)")
    plt.title("Pendulum Angular Velocity (Omega)")
    plt.legend()
    plt.grid()

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_file_path)
    print(f"Plot saved to {plot_file_path}")
    plt.close()


if __name__ == "__main__":
    # TODO: Add plotting functionality
    # Test parameters
    y0 = jnp.array(
        [jnp.pi / 4, 0.0]
    )  # Initial conditions: theta = 45 degrees, omega = 0.0
    t_span = (0.0, 10.0)  # Time interval: 0 to 10 seconds
    dt = 0.01  # Time step: 0.01 seconds
    b = 0.1  # Damping coefficient
    m = 1.0  # Mass of the pendulum
    l = 1.0  # Length of the pendulum
    g = 9.81  # Gravitational acceleration

    # Solve using Euler method
    t_euler, y_euler = solve_pendulum_euler(y0, t_span, dt, b, m, l, g)

    # Solve using Runge-Kutta 4th order method
    t_rk, y_rk = solve_pendulum_rk(y0, t_span, dt, b, m, l, g)

    # Plot the results
    plot_pendulum_results(
        t_euler,
        y_euler,
        t_rk,
        y_rk,
        save_dir="plots\generated_data",
        plot_filename="generated_data.png",
    )
