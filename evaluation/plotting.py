import matplotlib.pyplot as plt
import os
import numpy as np

def plot_each(data, name, save_dir):
    save_path = os.path.join(save_dir, "{}.png".format(name))
    if len(data) == 0:
        print(f"⚠️ No {name} found in model")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(data) + 1), data, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.title(f"{name} over Epochs")
    # Set x-axis ticks to integers
    xticks = plt.xticks()[0]  # Get current tick locations
    selected_ticks = np.linspace(xticks[0], xticks[-1], 5, dtype=np.int32)  # Select 5 evenly spaced ticks
    plt.xticks(selected_ticks, [int(tick) for tick in selected_ticks])  # Set the ticks
    plt.legend()
    plt.savefig(save_path)
    print(f"📉 {name} plot saved to {save_path}")

def plot_all(log_history, save_dir):
    save_plot_dir = os.path.join(save_dir, "plotting")
    os.makedirs(save_plot_dir, exist_ok=True)
    for key, item in log_history.items():
        plot_each(item, key, save_plot_dir)