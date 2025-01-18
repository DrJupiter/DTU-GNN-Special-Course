import wandb
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import uuid
import pickle
import jax
import jax.numpy as jnp
import optax
from pathlib import Path
from transformers import set_seed

from experiment.config import BuilderConfigExperiment
from data.dataloader import construct_dataloaders, compute_mean_std
from model.PaiNN import construct_PaiNN

# Generate a UUID v4 for this run, create results directory
run_id = str(uuid.uuid4())
results_dir = Path(f"./results/{run_id}")
results_dir.mkdir(parents=True, exist_ok=True)
wandb.init(project="GraphNN", name=run_id)

# Build and save config
set_seed(0)
config = (
    BuilderConfigExperiment()
    .set_vocab_dim(50)
    .set_feature_dim(128)
    .set_train_size(100_000)
    .set_validation_size(15_000)
    .set_batch_size(100)
    .set_num_workers(6)
    .set_radius(3.)
    .set_target_key("energy_U0")
    .set_path("./data/qm9.db")
    .set_split_file("./data/split3.npz")
    .build()
)

# Save the config to JSON
config.save(results_dir / "config.json")

# Construct data loaders
train_dl, test_dl, val_dl = construct_dataloaders(config.train)

#MEAN, STD = compute_mean_std(train_dl, config.train.target_key)

MEAN, STD = -410.8194, 39.8941

NORMALIZE = lambda data: (data - MEAN) / STD

# Grab a single batch just to ensure everything works (optional for debugging)
example_data_point = next(iter(train_dl))
positions = jax.numpy.array(example_data_point["_positions"].numpy())
atomic_numbers = jax.numpy.array(example_data_point["_atomic_numbers"].numpy())
idx_i = jax.numpy.array(example_data_point["_idx_i"].numpy())
idx_j = jax.numpy.array(example_data_point["_idx_j"].numpy())
idx_m = jax.numpy.array(example_data_point["_idx_m"].numpy())

# s0 and v0
s0 = atomic_numbers[:, jnp.newaxis]
v0 = jnp.zeros((len(atomic_numbers), 3, config.model.feature_dim))

# Construct the model and initialize parameters
model, params = construct_PaiNN(
    vocab_dim=config.model.vocab_dim,
    feature_dim=config.model.feature_dim,
)

# Optimizer

#learning_rate = 1e-3

learning_rate_fn = optax.exponential_decay(
    init_value=5e-4,        # Initial learning rate
    transition_steps=1000,  # How many steps before applying decay
    decay_rate=0.99,        # Rate of exponential decay (e.g., 0.99 ~ 1% decay)
    staircase=True          # Whether to apply discrete "staircase" decay
)


base_optimizer = optax.adamaxw(learning_rate_fn)
ema_transform = optax.ema(decay=0.999)
grad_clipping = optax.clip_by_global_norm(max_norm=1.0)

optimizer = optax.chain(
    grad_clipping,
    ema_transform,
    base_optimizer
)
opt_state = optimizer.init(params)



def train_step(params, opt_state, v0, s0, directions, idx_i, idx_j, idx_m, target):
    """Compute forward pass, loss, grads, then update via Adam."""

    def loss_fn(p):
        predictions = model(p, v0, s0, directions, idx_i, idx_j, idx_m)
        return jnp.mean((predictions - target) ** 2)

    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

LEARNING_RATE = 5e-4
def train_step_sgd(params, opt_state, v0, s0, directions, idx_i, idx_j, idx_m, target):
    """Compute forward pass, loss, grads, then update via Adam."""

    def loss_fn(p):
        predictions = model(p, v0, s0, directions, idx_i, idx_j, idx_m)
        return jnp.mean((predictions - target) ** 2)

    grads = jax.grad(loss_fn)(params)
    new_params = jax.tree.map(
        lambda p, g: p - LEARNING_RATE * g, params, grads
    )
    return new_params, ()




# Training loop
num_epochs = 3
print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for batch_idx, data_point in enumerate(train_dl):
        # Extract batch
        positions = jax.numpy.array(data_point["_positions"].numpy())
        atomic_numbers = jax.numpy.array(data_point["_atomic_numbers"].numpy())
        idx_i = jax.numpy.array(data_point["_idx_i"].numpy())
        idx_j = jax.numpy.array(data_point["_idx_j"].numpy())
        idx_m = jax.numpy.array(data_point["_idx_m"].numpy())

        # Construct s0, v0, and directions
        s0 = atomic_numbers[:, jnp.newaxis]
        v0 = jnp.zeros((len(atomic_numbers), 3, config.model.feature_dim))
        directions = positions[idx_j] - positions[idx_i]

        # Fetch target (e.g. property to predict)
        target = jax.numpy.array(NORMALIZE(data_point[config.train.target_key]).numpy())

        # Perform training step
        params, opt_state = train_step(
            params,
            opt_state,
            v0,
            s0,
            directions,
            idx_i,
            idx_j,
            idx_m,
            target
        )

        # Logging
        if batch_idx % 1 == 0:
            train_loss = jnp.mean(
                (model(params, v0, s0, directions, idx_i, idx_j, idx_m) - target)**2
            )
            wandb.log({"loss": train_loss})
            print(f"  [Batch {batch_idx}] MSE Loss: {train_loss:.6f}")

# Save final trained model parameters
model_save_path = results_dir / "model_params.pkl"
with open(model_save_path, "wb") as f:
    pickle.dump(params, f)

wandb.finish()
print(f"\nTraining complete. Model parameters saved to {model_save_path}")
print(f"Config saved to {results_dir / 'config.json'}")
print("Run ID:", run_id)
