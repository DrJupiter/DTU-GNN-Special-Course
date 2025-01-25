import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_mean_ci(directory, paper_json_path=None):
    # Define file paths
    not_normalized_file = os.path.join(directory, "not_normalized.json")
    normalized_file = os.path.join(directory, "normalized.json")

    # Check if files exist
    if not os.path.exists(not_normalized_file) or not os.path.exists(normalized_file):
        print("Required files not found in the specified directory.")
        return

    data = {}
    # Load data
    if os.path.exists(not_normalized_file):
        with open(not_normalized_file, "r") as f:
            not_normalized_data = json.load(f)
        if len(not_normalized_data) != 0:
            data.update({"Ours": not_normalized_data})
    else:
        not_normalized_data = None

    if os.path.exists(normalized_file):
        with open(normalized_file, "r") as f:
            normalized_data = json.load(f)
        if len(normalized_data) != 0:
            data.update({"Ours*": normalized_data})
    else:
        normalized_data = None

    # Calculate mean, standard error, and 99% confidence interval
    means = {key: np.mean(values) for key, values in data.items()}
    stds = {key: np.std(values, ddof=1) for key, values in data.items()}  # Use ddof=1 for sample std
    sizes = {key: len(values) for key, values in data.items()}

    z_value = 2.576  # Z-value for 99% confidence interval
    ci_values = {key: z_value * (stds[key] / np.sqrt(sizes[key])) for key in data.keys()}
    lower_bounds = {key: means[key] - ci_values[key] for key in data.keys()}
    upper_bounds = {key: means[key] + ci_values[key] for key in data.keys()}

    # Add paper values if provided
    if paper_json_path is not None and os.path.exists(paper_json_path):
        with open(paper_json_path, "r") as f:
            paper_data = json.load(f)
        for paper_name, paper_mean in paper_data.items():
            data[paper_name] = [paper_mean]
            means[paper_name] = paper_mean
            ci_values[paper_name] = 0  # No CI since std is not provided
            lower_bounds[paper_name] = paper_mean
            upper_bounds[paper_name] = paper_mean

    # Sort data by mean values in descending order
    sorted_items = sorted(means.items(), key=lambda item: item[1], reverse=True)
    sorted_labels = [item[0] for item in sorted_items]
    sorted_means = [item[1] for item in sorted_items]
    sorted_ci_errors = [ci_values[label] for label in sorted_labels]

    # Define colors
    our_results_labels = ["Ours*",
        "Ours"]
    paper_results_labels = [label for label in sorted_labels if label not in our_results_labels]

    # Color maps
    our_cmap = plt.get_cmap('Blues')
    paper_cmap = plt.get_cmap('Greens')
    highlight_color = '#FF6347'  # Tomato color for highlighting

    # Assign colors to bars
    bar_colors = []
    for label in sorted_labels:
        if label in our_results_labels:
            bar_colors.append(our_cmap(0.6 if label == "Not Normalized" else 0.9))
        else:
            bar_colors.append(paper_cmap(0.6))

    # Highlight the bar with the lowest mean value
    min_index = sorted_means.index(min(sorted_means))
    bar_colors[min_index] = highlight_color

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.bar(
        sorted_labels,
        sorted_means,
        yerr=sorted_ci_errors,
        capsize=8,
        alpha=0.85,
        color=bar_colors,
        edgecolor="black"
    )

    # Annotate each bar with mean, lower, and upper bound
    for i, bar in enumerate(bars):
        mean = sorted_means[i]
        ci = sorted_ci_errors[i]
        lower = mean - ci
        upper = mean + ci
        annotation = f"Mean: {mean:.2f}"
        if ci > 0:
            annotation += f"\nLB: {lower:.2f}, UB: {upper:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ci if ci > 0 else 0.2),
            annotation,
            ha='center',
            fontsize=12,
            color="#333333"
        )

    # Add labels and title
    ax.set_title("Mean and 99% Confidence Interval of MAE", fontsize=20, fontweight="bold", pad=20)
    ax.set_ylabel("Mean Absolute Error", fontsize=16, labelpad=10)
    ax.set_xlabel("Models", fontsize=16, labelpad=10)
    ax.set_ylim(0, max(sorted_means) + max(sorted_ci_errors) + 1)

    # Enhance tick formatting and grid lines
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle='--', alpha=0.6, color="#b0b0b0")

    # Add custom style
    plt.xticks(fontsize=12, fontweight="bold", rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Show plot
    plt.show()

if __name__ == "__main__":
    import sys
    directory = sys.argv[1]
    paper_json_path = sys.argv[2] if len(sys.argv) > 2 else None
    plot_mean_ci(directory, paper_json_path)
