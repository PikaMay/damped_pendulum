import jax
import jax.numpy as jnp
from flax.training.early_stopping import EarlyStopping
from data_generator import solve_pendulum_rk
from train import MLP, create_train_state
import matplotlib.pyplot as plt
import os
import hydra
from omegaconf import DictConfig


def ode_loss(params, apply_fn, batch, ode_params=(0.3, 1.0, 1.0, 9.81)):

    t, _ = batch  # extract time values from the batch
    b, m, l, g = ode_params

    # Vectorize apply_fn over the batch dimension
    def compute_loss_for_t(t):
        # Compute predicted theta (MLP output) for given time t
        theta_pred = apply_fn(params, t)

        # Compute the first derivative of theta (angular velocity)
        theta_dot = jax.grad(lambda t: jnp.sum(apply_fn(params, t)))(t)

        # Compute the second derivative of theta (angular acceleration)
        theta_ddot = jax.grad(lambda t: jnp.sum(theta_dot))(t)

        # Compute the ODE residual
        ode_residual = (
            theta_ddot + (b / (m * l)) * theta_dot + (g / l) * jnp.sin(theta_pred)
        )

        # Compute the initial condition loss for the angle at t=0
        angle_loss = (
            theta_pred[0] - 2 * jnp.pi / 3
        ) ** 2  # Target initial angle is 2*pi/3

        # Compute the initial condition loss for the angular velocity at t=0
        angular_velocity_loss = (
            theta_dot[0]
        ) ** 2  # Assuming initial angular velocity is 0

        # Combine all terms to get the total loss
        total_loss = jnp.mean(ode_residual**2) + angle_loss + angular_velocity_loss

        return total_loss

    vectorized_loss = jax.vmap(compute_loss_for_t)(
        t
    )  # Apply the loss function to each time entry

    return jnp.mean(vectorized_loss)  # Average the loss over the batch

    pass  # TODO: Complete this function


@jax.jit
def ode_train_step(state, batch):
    """A train step using the ode_loss."""

    def loss_fn(params):
        return ode_loss(params, state.apply_fn, batch)

    loss = loss_fn(state.params)

    # Compute the loss and gradients
    grads = jax.grad(loss_fn)(state.params)

    # Update the model parameters using the optimizer
    state = state.apply_gradients(grads=grads)

    # Updating  the metrics in the sate
    state = state.replace(metrics={"loss": loss})

    return state
    pass  # TODO: Complete this function


def ode_train(key, model, learning_rate, epochs, data):
    state = create_train_state(model, key, learning_rate, (1,))
    ode_metrics_history = {
        "ode_loss": [],
    }
    t, y = data
    early_stop = EarlyStopping(min_delta=1e-3, patience=2)
    for epoch in range(epochs):
        # Training step
        batch = (t[:, None], y[:, 0][:, None])
        state = ode_train_step(state, batch)
        loss = state.metrics["loss"]
        early_stop = early_stop.update(loss)

        # Store metrics
        ode_metrics_history["ode_loss"].append(loss)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, ODE Loss: {loss}")

            # Check for early stopping
        if early_stop.should_stop:
            print("Met early stopping criteria, breaking...")
            break

    return state, ode_metrics_history


def plot_loss(
    ode_metrics_history, save_dir="./loss_plots", plot_filename="ode_loss_plot.png"
):
    """
    Function to plot and save the loss curve.

    Parameters:
    - ode_metrics_history (dict): The dictionary containing the loss values.
    - save_dir (str): Directory to save the plot.
    - plot_filename (str): The filename to save the plot.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create the full path to save the plot
    plot_file_path = os.path.join(save_dir, plot_filename)

    # Plotting the loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(ode_metrics_history["ode_loss"], label="ODE Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(plot_file_path)
    print(f"Plot saved to {plot_file_path}")

    # Display the plot
    plt.show()


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    # Extract dataset parameters from the config file
    y0 = jnp.array(cfg.dataset.y0)
    t_span = tuple(cfg.dataset.t_span)
    dt = cfg.dataset.dt
    b, m, l, g = cfg.dataset.b, cfg.dataset.m, cfg.dataset.l, cfg.dataset.g
    data = solve_pendulum_rk(y0, t_span, dt, b, m, l, g)

    key = jax.random.PRNGKey(cfg.train.key)
    model = MLP(cfg.model.features)
    learning_rate = cfg.train.learning_rate
    epochs = cfg.train.epochs
    state, ode_metrics_history = ode_train(key, model, learning_rate, epochs, data)

    os.makedirs(cfg.output_dirs.ode_loss, exist_ok=True)
    plot_loss(
        ode_metrics_history,
        save_dir=cfg.output_dirs.ode_loss,
        plot_filename="ode_loss_plot.png",
    )


if __name__ == "__main__":
    main()

# TODO: Add plotting functionality
