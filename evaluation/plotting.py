import matplotlib.pyplot as plt
import os
import numpy as np

def plot_each(train_data, val_data, name, save_dir):
    save_path = os.path.join(save_dir, f"{name}.png")

    if len(train_data) == 0 and len(val_data) == 0:
        print(f"⚠️ No data found for {name}")
        return

    plt.figure(figsize=(8, 5))
    
    if len(train_data) > 0:
        plt.plot(range(1, len(train_data) + 1), train_data, label=f"train_{name}")
    if len(val_data) > 0:
        plt.plot(range(1, len(val_data) + 1), val_data, label=f"val_{name}")

    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.title(f"{name} over Epochs")

    xticks = plt.xticks()[0]
    selected_ticks = np.linspace(xticks[0], xticks[-1], 5, dtype=np.int32)
    plt.xticks(selected_ticks, [int(tick) for tick in selected_ticks])

    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"📉 {name} plot saved to {save_path}")

def plot_all(log_history, save_dir):
    save_plot_dir = os.path.join(save_dir, "plotting")
    os.makedirs(save_plot_dir, exist_ok=True)

    keys = list(log_history.keys())
    metric_names = set(k.replace("train_", "").replace("val_", "") for k in keys)

    for name in metric_names:
        train_data = log_history.get(f"train_{name}", [])
        val_data = log_history.get(f"val_{name}", [])
        plot_each(train_data, val_data, name, save_plot_dir)
