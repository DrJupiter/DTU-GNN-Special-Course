# DTU-GNN-Special-Course

### Training, Evaluating, and Plotting Models

This document provides a guide to use the `train_models.sh` script for training machine learning models, followed by evaluating and plotting the results.

---

#### Prerequisites
Before running the scripts, ensure the following are installed:
1. **Python** (Version >= 3.6) with the required libraries for `torch_main.py`, `torch_evaluate.py`, and `plot_results.py`.
2. **Bash Shell** for executing the `train_models.sh` script.
3. Ensure `torch_main.py`, `torch_evaluate.py`, and `plot_results.py` are in the same directory as `train_models.sh` or provide the correct paths to them.

---

### Step 1: Train Models

#### Usage

The `train_models.sh` script is used to train multiple machine learning models concurrently.

```bash
./train_models.sh [TARGET_VARIABLE] [NUM_MODELS]
```

#### Arguments
- `TARGET_VARIABLE`: The target variable for model training (e.g., `energy_U0`). Defaults to `"energy_U0"` if not provided.
- `NUM_MODELS`: The number of models to train. Defaults to `5` if not provided.

#### Example
To train models with the target variable `dipole_moment` and train 10 models:
```bash
./train_models.sh dipole_moment 10
```

#### Notes
- The script runs with a maximum of 2 concurrent processes (`MAX_JOBS`).
- Models are trained using `torch_main.py` with two modes (indicated by `false` and `true` for each model).

---

### Step 2: Evaluate Results

After training the models, evaluate the results using the following command:

```bash
python3 torch_evaluate.py ./results/[TARGET_VARIABLE]/
```

Replace `[TARGET_VARIABLE]` with the directory where the results are stored. For example:
```bash
python3 torch_evaluate.py ./results/dipole_moment/
```

---

### Step 3: Plot Results

Finally, generate visualizations of the results using:

```bash
python3 plot_results.py ./results/[TARGET_VARIABLE]/
```

For example:
```bash
python3 plot_results.py ./results/dipole_moment/
```

---

### Example Workflow

Here’s a complete example workflow for training, evaluating, and plotting models with `dipole_moment` as the target variable and training 10 models:

```bash
# Step 1: Train Models
./train_models.sh dipole_moment 10

# Step 2: Evaluate Results
python3 torch_evaluate.py ./results/dipole_moment/

# Step 3: Plot Results
python3 plot_results.py ./results/dipole_moment/
```

---

### Additional Notes
- Ensure the `results` directory structure matches the target variable (`./results/[TARGET_VARIABLE]/`).
- Logs and errors during training will be displayed in the terminal. Modify the script to save logs if needed.
- Adjust the `MAX_JOBS` variable in the `train_models.sh` script to control the number of concurrent processes.

For any issues, check the error messages or ensure the required Python libraries are installed.
