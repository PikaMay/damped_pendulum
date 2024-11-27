import jax
import jax.numpy as jnp
from jax import grad, vmap
from data_generator import solve_pendulum_rk
from train import MLP, create_train_state
import matplotlib.pyplot  as plt
import os


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
        ode_residual = theta_ddot + (b / (m * l)) * theta_dot + (g / l) * jnp.sin(theta_pred)
        
        # Compute the initial condition loss for the angle at t=0
        angle_loss = (theta_pred[0] - 2 * jnp.pi / 3)**2  # Target initial angle is 2*pi/3
        
        # Compute the initial condition loss for the angular velocity at t=0
        angular_velocity_loss = (theta_dot[0])**2  # Assuming initial angular velocity is 0
        
        # Combine all terms to get the total loss
        total_loss = jnp.mean(ode_residual**2) + angle_loss + angular_velocity_loss
        
        return total_loss
    
    # Use vmap to apply the loss function over the entire batch
    vectorized_loss = jax.vmap(compute_loss_for_t)(t)  # Apply the loss function to each time entry
    
    # Return the mean loss across all time steps
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
    for epoch in range(epochs):
        # Training step
        batch = (t[:, None], y[:, 0][:, None])
        state = ode_train_step(state, batch)
        loss = state.metrics["loss"]

        # Store metrics
        ode_metrics_history["ode_loss"].append(loss)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, ODE Loss: {loss}")

    return state, ode_metrics_history

def plot_loss(ode_metrics_history, save_dir='./loss_plots', plot_filename='ode_loss_plot.png'):
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
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(plot_file_path)
    print(f"Plot saved to {plot_file_path}")

    # Display the plot
    plt.show()



if __name__ == "__main__":
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])
    t_span = (0, 20)
    dt = 0.01
    b = 0.3
    m = 1.0
    l = 1.0
    g = 9.81
    data = solve_pendulum_rk(y0, t_span, dt, b, m, l, g)
    key = jax.random.PRNGKey(0)
    model = MLP([16, 16, 16])
    learning_rate = 1e-3
    epochs = 100_000
    state, ode_metrics_history = ode_train(key, model, learning_rate, epochs, data)

    # TODO: Add plotting functionality

    plot_loss(ode_metrics_history, save_dir='./plots', plot_filename='ode_loss_plot.png')
