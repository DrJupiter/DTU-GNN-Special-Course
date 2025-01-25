import wandb
import uuid
import torch
import torch.optim as optim
import torch.nn.utils as utils
from pathlib import Path
from transformers import set_seed

from experiment.config import BuilderConfigExperiment
from data.dataloader import construct_dataloaders, compute_mean_std, prepare_data_point
from torch_model.PaiNN import PaiNN
import sys

args = sys.argv[1:]

TARGET = args[0].strip()
SEED = int(args[1].strip())
NORM = args[2].strip().lower() == "true"

# 1. Generate a UUID v4 for this run, create results directory
run_id = str(uuid.uuid4())
results_dir = Path(f"./results/{TARGET}/{run_id}")
results_dir.mkdir(parents=True, exist_ok=True)

wandb.init(project="GNN", name=run_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Build and save config
set_seed(SEED)
config = (
    BuilderConfigExperiment()
    .set_vocab_dim(20)
    .set_feature_dim(128)
    .set_train_size(100_000)
    .set_validation_size(15_000)
    .set_batch_size(100)
    .set_num_workers(1)
    .set_radius(3.)
    .set_seed(SEED)
    .set_target_key(TARGET)
    .set_normalize(NORM)
    .set_path("./data/qm9.db")
    .set_split_file(f"./data/splits/split_{SEED}.npz")
    .build()
)

config = config.set_device(str(device))

# Save the config to JSON
config.save(results_dir / "config.json")

#  Construct data loaders
train_dl, test_dl, val_dl = construct_dataloaders(config.train)

#MEAN, STD = (torch.tensor(-410.7464), torch.tensor(39.9485))
if config.train.normalize:
    MEAN, STD = compute_mean_std(train_dl, config.train.target_key)
    normalize = lambda data: (data-MEAN)/STD
    denormalize = lambda prediction: (prediction*STD) + MEAN
else:
    normalize = lambda data: data
    denormalize = lambda prediction: prediction

#  Construct the model and initialize parameters
model = PaiNN(
    vocab_dim=config.model.vocab_dim,
    feature_dim=config.model.feature_dim,
)

model = model.to(config.model.device)

# Initial learning rate
initial_lr = 5e-4
#initial_lr = 1e-5

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=initial_lr,  weight_decay=1e-5)
#optimizer = optim.AdamW(model.parameters(), lr=initial_lr)

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
num_epochs = 100
print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for batch_idx, data_point in enumerate(train_dl):

        optimizer.zero_grad()

        input_data, target = prepare_data_point(data_point, config.model.feature_dim ,config.train.target_key, config.model.device, normalize)

        # Perform training step
        prediction = model.forward(**input_data)

        loss = torch.nn.functional.mse_loss(prediction, target)

        if loss.item() == 0.0 or torch.isnan(loss):
            print(f"Skipping update due to invalid loss: {loss.item()}")
            continue  # Skip this batch

        loss.backward()

        # Apply gradient clipping
        utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        # Update EMA parameters
        update_ema(model, ema_params, ema_decay)

        if batch_idx % 5 == 0:
            loss = torch.nn.functional.mse_loss(denormalize(prediction.detach().cpu()), denormalize(target.detach().cpu()))
            wandb.log({"Train Loss": loss.detach().cpu().item()})
        if batch_idx % 10 == 0:
            model.eval()
            data_point = next(iter(val_dl))
            input_data, target = prepare_data_point(data_point, config.model.feature_dim , config.train.target_key, config.model.device, normalize)
            prediction = model.forward(**input_data)
            loss = torch.nn.functional.mse_loss(denormalize(prediction.detach().cpu()), denormalize(target.detach().cpu()))
            wandb.log({"Validation Loss": loss.detach().cpu().item()})
            model.train()


    scheduler.step()

# 8. Save final trained model parameters
model_save_path = results_dir / "model_params.pt"
torch.save(model.state_dict(), model_save_path)
wandb.finish()

print(f"\nTraining complete. Model parameters saved to {model_save_path}")
print(f"Config saved to {results_dir / 'config.json'}")
print("Run ID:", run_id)
