import sys
import torch
from pathlib import Path
from transformers import set_seed

# Import your own classes/modules
from experiment.config import ConfigExperiment
from data.dataloader import construct_dataloaders, compute_mean_std, prepare_data_point
from torch_model.PaiNN import PaiNN
import json

def main():
    """
    Usage:
        python load_models_by_group.py /path/to/results/energy_U0
    """
    if len(sys.argv) < 2:
        print("Usage: python load_models_by_group.py <results_base_dir>")
        sys.exit(1)

    base_dir = Path(sys.argv[1].strip())

    # Lists for storing subdirectories that used normalization vs. not
    normalized_runs = []
    not_normalized_runs = []

    # 1. Identify each run folder under base_dir
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue  # skip if not a directory

        config_path = run_dir / "config.json"
        model_params_path = run_dir / "model_params.pt"

        # 2. Check if both files exist; if so, load them
        if config_path.exists() and model_params_path.exists():
            # Load config using your from_pretrained method
            config = ConfigExperiment.from_pretrained(config_path)

            # Sort runs based on the "normalize" flag
            if config.train.normalize:
                normalized_runs.append(run_dir)
            else:
                not_normalized_runs.append(run_dir)
        else:
            # If missing either file, skip or handle differently
            print(f"Skipping {run_dir} - missing config/model_params.")

    print("=== Normalized Runs ===")
    for run_dir in normalized_runs:
        print(f" • {run_dir}")

    print("=== Not Normalized Runs ===")
    for run_dir in not_normalized_runs:
        print(f" • {run_dir}")
    # Optional: show how you might loop and load the model
    #           for each group in detail
    print("\nLoading Normalized Runs...")
    normalized_losses = []
    for run_dir in normalized_runs:
        config_path = run_dir / "config.json"
        model_params_path = run_dir / "model_params.pt"

        config = ConfigExperiment.from_pretrained(config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (Optional) Re-seed
        if hasattr(config.train, "seed"):
            set_seed(config.train.seed)

        # Construct model
        model = PaiNN(
            vocab_dim=config.model.vocab_dim,
            feature_dim=config.model.feature_dim,
        ).to(device)

        # Load model weights
        model.load_state_dict(torch.load(model_params_path, map_location=device))
        model.eval()
        print(f"  Loaded normalized model from {run_dir}")

        # (Optional) If you need data loaders or mean/std:
        #if config.train.normalize:
        train_dl, test_dl, val_dl = construct_dataloaders(config.train)
        mean, std = compute_mean_std(train_dl, config.train.target_key)
            # e.g. run inference/validation here...
        normalize = lambda data: (data-mean)/std
        denormalize = lambda prediction: (prediction * std) + mean

        normalized_losses.append(evaluate_model(model, test_dl, config, normalize, denormalize))
    save_results(base_dir / "normalized.json", normalized_losses)

    print("\nLoading Not Normalized Runs...")
    not_normalized_losses = []
    for run_dir in not_normalized_runs:
        config_path = run_dir / "config.json"
        model_params_path = run_dir / "model_params.pt"

        config = ConfigExperiment.from_pretrained(config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (Optional) Re-seed
        if hasattr(config.train, "seed"):
            set_seed(config.train.seed)

        # Construct model
        model = PaiNN(
            vocab_dim=config.model.vocab_dim,
            feature_dim=config.model.feature_dim,
        ).to(device)

        # Load model weights
        model.load_state_dict(torch.load(model_params_path, map_location=device))
        model.eval()
        print(f"  Loaded not-normalized model from {run_dir}")

        train_dl, test_dl, val_dl = construct_dataloaders(config.train)
        normalize = lambda data: data
        denormalize = lambda prediction: prediction
        not_normalized_losses.append(evaluate_model(model, test_dl, config, normalize, denormalize))

    save_results(base_dir / "not_normalized.json", not_normalized_losses)

def evaluate_model(model, dataloader, config: ConfigExperiment, normalize, denormalize) -> float:

    losses: list[torch.Tensor] = []
    with torch.inference_mode():
        for data in dataloader:
            input_data, target = prepare_data_point(data, config.model.feature_dim , config.train.target_key, config.model.device, normalize)
            prediction = model.forward(**input_data)
            loss = torch.nn.functional.l1_loss(denormalize(prediction.detach().cpu()), denormalize(target.detach().cpu()), reduction='none')
            valid_loss = loss[~torch.isnan(loss)]
            losses.extend(valid_loss)

    return float(torch.stack(losses).mean().item())

def save_results(path: str | Path, results: list[float]):
    with open(path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
