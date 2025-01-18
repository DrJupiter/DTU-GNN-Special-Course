import wandb
import uuid
import pickle
import torch
import torch.optim as optim
import torch.nn.utils as utils
from pathlib import Path
from transformers import set_seed

from experiment.config import BuilderConfigExperiment
from data.dataloader import construct_dataloaders, compute_mean_std
from torch_model.PaiNN import PaiNN

# 1. Generate a UUID v4 for this run, create results directory
run_id = str(uuid.uuid4())
results_dir = Path(f"./results/{run_id}")
results_dir.mkdir(parents=True, exist_ok=True)

wandb.init(project="GNN", name=run_id)

# 2. Build and save config
set_seed(0)
config = (
    BuilderConfigExperiment()
    .set_vocab_dim(20)
    .set_feature_dim(128)
    .set_train_size(100_000)
    .set_validation_size(15_000)
    .set_batch_size(100)
    .set_num_workers(2)
    .set_radius(3.)
    .set_path("./data/qm9.db")
    .set_split_file("./data/split2.npz")
    .build()
)

# Save the config to JSON
config.save(results_dir / "config.json")

#  Construct data loaders
train_dl, test_dl, val_dl = construct_dataloaders(config.train)

#MEAN, STD = compute_mean_std(train_dl, "energy_U0")
MEAN, STD = (torch.tensor(-410.7464), torch.tensor(39.9485))
normalize = lambda data: (data-MEAN)/STD

#  Construct the model and initialize parameters
model = PaiNN(
    vocab_dim=config.model.vocab_dim,
    feature_dim=config.model.feature_dim,
)

# 5. Set up an Adam optimizer with Optax
#learning_rate = 1e-3

# Initial learning rate
initial_lr = 5e-4

# Initialize the optimizer
#optimizer = optim.Adam(model.parameters(), lr=initial_lr)
optimizer = optim.AdamW(model.parameters(), lr=initial_lr)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Initialize EMA parameters
ema_decay = 0.999
ema_params = {name: param.clone().detach() for name, param in model.named_parameters()}

# Function to update EMA parameters
def update_ema(model, ema_params, decay):
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)

# 7. Training loop
num_epochs = 3
print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for batch_idx, data_point in enumerate(train_dl):

        optimizer.zero_grad()

        # Extract batch
        positions = data_point["_positions"]
        atomic_numbers = data_point["_atomic_numbers"]
        idx_i = data_point["_idx_i"]
        idx_j = data_point["_idx_j"]
        idx_m = data_point["_idx_m"]

        # Construct s0, v0, and directions
        s0 = atomic_numbers[:, None]
        v0 = torch.zeros((len(atomic_numbers), 3, config.model.feature_dim))
        directions = positions[idx_j] - positions[idx_i]

        # Fetch target (e.g. property to predict)
        target = normalize(data_point["energy_U0"])

        # Perform training step
        prediction = model.forward(
            v0,
            s0,
            directions,
            idx_i,
            idx_j,
            idx_m,
        )

        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()

        # Apply gradient clipping
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update EMA parameters
        update_ema(model, ema_params, ema_decay) 

        wandb.log({"Train Loss": loss.item()})
        
        # (Optional) Logging
        #if batch_idx % 1 == 0:
        #    print(f"  [Batch {batch_idx}] MSE Loss: {loss:.6f}")

# 8. Save final trained model parameters
model_save_path = results_dir / "model_params.pt"
torch.save(model.state_dict(), model_save_path)
wandb.finish()

print(f"\nTraining complete. Model parameters saved to {model_save_path}")
print(f"Config saved to {results_dir / 'config.json'}")
print("Run ID:", run_id)
