import flax.linen as nn
import jax
import optax
import jax.numpy as jnp
from flax.training import train_state
from dataclasses import replace
import matplotlib.pyplot as plt
from data_generator import gen_data, solve_pendulum_rk
import os 

class MLP(nn.Module):
    """Multilayer Perceptron model."""

    features: list[int]

    def setup(self):
        """Setup layers based on the features sizes """
        self.layers = []
        self.layers = [nn.Dense(feat) for feat in self.features]

    
    def __call__(self, x):
        """Forward pass of the MLP"""

        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"Layer {i} output shape: {x.shape}")
            if i != len(self.layers) - 1 :
                x = nn.relu(x)
        return x 

    #    pass  # TODO: Complete this class


class TrainState(train_state.TrainState):
    metrics: dict


def create_train_state(model, init_key, learning_rate, input_shape):
    """Initialize the train state for the model."""

    # Initialize the model parameters
    params = model.init(init_key, jnp.ones(input_shape))

    # Define the Adap Optimizer
    tx = optax.adam(learning_rate)

    # TrainState Object
    state = TrainState.create(apply_fn=model.apply, tx=tx, params=params, metrics={}) 

    return state
    pass  # TODO: Complete this function

def mse_loss(params, apply_fn, batch):
    """Creates a MSE loss function"""

    t, y_true = batch

    y_pred = apply_fn(params, t) # Forward pass

    return jnp.mean((y_pred - y_true) ** 2)
    pass  # TODO: Complete this function


@jax.jit
def compute_metrics(state, batch):
    loss = mse_loss(state.params, state.apply_fn, batch)
    return {"loss": loss}
    pass  # TODO: Complete this function


@jax.jit
def train_step(state, batch):
    """ A single training step"""

    def loss_fn(params):
        return mse_loss(params, state.apply_fn, batch)
    

    grads = jax.grad(loss_fn)(state.params)  # Compute gradients of the loss function
    state = state.apply_gradients(grads=grads)  # Update model parameters using computed gradients
    metrics = compute_metrics(state, batch)  # Compute metrics 
    state = replace(state, metrics=metrics)

    return state
    pass  # TODO: Complete this function


@jax.jit
def val_step(state, batch):
    metrics = compute_metrics(state, batch)
    return metrics
    pass  # TODO: Complete this function


def train(key, model, learning_rate, epochs, data):
    t_train, y_train, t_test, y_test = data
    state = create_train_state(model, key, learning_rate, (1,))
    metrics_history = {
        "train_loss": [],
        "test_loss": [],
    }
    for epoch in range(epochs):
        # Training step
        train_batch = (t_train[:, None], y_train[:, None])
        state = train_step(state, train_batch)
        train_loss = state.metrics["loss"]

        # Validation step
        val_batch = (t_test[:, None], y_test[:, None])
        val_loss = val_step(state, val_batch)



        # Store metrics
        metrics_history["train_loss"].append(train_loss)
        metrics_history["test_loss"].append(val_loss)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
    return state, metrics_history

def plot_train_and_test_losses(metrics_history, save_dir=''):
    """Plot training loss and test loss as separate plots and save them."""
    
    # Extract the losses
    train_losses = [loss.item() for loss in metrics_history['train_loss']]  # Convert train_loss values to scalars
    test_losses = [entry['loss'].item() for entry in metrics_history['test_loss']]  # Convert test_loss values to scalars

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color='blue', linestyle='-', marker='o', markersize=3)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    train_save_path = os.path.join(save_dir, 'train_loss.png')
    plt.savefig(train_save_path)
    print(f"Training loss plot saved to {train_save_path}")
    plt.show()

    # Plot test loss
    plt.figure(figsize=(10, 6))
    plt.plot(test_losses, label="Test Loss", color='red', linestyle='--', marker='x', markersize=3)
    plt.title("Test Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    test_save_path = os.path.join(save_dir, 'test_loss.png')
    plt.savefig(test_save_path)
    print(f"Test loss plot saved to {test_save_path}")
    plt.show()

def plot_predictions(t, y_true, y_pred, save_dir=''):
    """
    Plots the model's predictions vs. the ground truth.

    Args:
        t: Time points.
        y_true: Ground truth ODE solutions.
        y_pred: Model predictions.
        save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_true[:, 0], label="True θ (Ground Truth)", linestyle="--", color="blue")
    plt.plot(t, y_pred[:, 0], label="Predicted θ", linestyle="-", color="orange")
    plt.xlabel("Time (t)")
    plt.ylabel("θ (Angle)")
    plt.title("Predictions vs. Ground Truth")
    plt.legend()
    if save_dir:
        plt.savefig(save_dir)
    plt.show()


if __name__ == "__main__":
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])
    t_span = (0, 20)
    dt = 0.01
    b = 0.3
    m = 1.0
    l = 1.0
    g = 9.81
    t, y = solve_pendulum_rk(y0, t_span, dt, b, m, l, g)
    data = gen_data(t, y)

    key = jax.random.PRNGKey(0)
    model = MLP([16, 16, 16])
    learning_rate = 1e-3
    epochs = 100_000

    state, metrics_history = train(key, model, learning_rate, epochs, data)
    
    # TODO: Add plotting functionality


    t = t[:, None]  # Reshape to add a batch dimension (2000, 1)

    # Generate predictions using vectorized mapping
    y_pred = jax.vmap(state.apply_fn, in_axes=(None, 0))(state.params, t)
    print(f"y_pred are : {y_pred}")
    print(f"the shape is : {y_pred.shape}")

 
    plot_train_and_test_losses(metrics_history, save_dir='./loss_plots')
    # Plot predictions vs. ground truth
    plot_predictions(t.flatten(), y, y_pred, save_dir="plots/predictions_vs_ground_truth.png")