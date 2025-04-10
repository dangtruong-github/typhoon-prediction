import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

def extract_metrics_from_event_file(event_file, metrics_to_extract):
    """Extract specified metrics from a TensorFlow event file."""
    ea = event_accumulator.EventAccumulator(
        path=event_file
    )
    ea.Reload()
    
    available_tags = ea.Tags()['scalars']
    result = {}
    
    for metric in metrics_to_extract:
        if metric in available_tags:
            events = ea.Scalars(metric)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            result[metric] = {'steps': steps, 'values': values}
    
    return result

def plot_metrics_from_folder(folder_path, metrics_to_plot):
    print(folder_path)

    fig_save_folder_path = os.path.join("/".join(folder_path.split("/")[:-2]), "plotting_partial")
    
    print(fig_save_folder_path)

    """Plot metrics from all event files in a folder."""
    all_event_files = []
    
    # Find all event files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith("events.out.tf"):
                all_event_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_event_files)} event files")
    
    # Extract metrics from each file
    all_metrics = {}
    for event_file in all_event_files:
        metrics = extract_metrics_from_event_file(event_file, metrics_to_plot)
        
        # Merge with existing metrics
        for metric_name, metric_data in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = {'steps': [], 'values': []}
            
            all_metrics[metric_name]['steps'].extend(metric_data['steps'])
            all_metrics[metric_name]['values'].extend(metric_data['values'])
    
    # Sort metrics by step (in case they're out of order)
    for metric_name in all_metrics:
        steps = np.array(all_metrics[metric_name]['steps'])
        values = np.array(all_metrics[metric_name]['values'])
        
        # Sort by step
        sort_idx = np.argsort(steps)
        all_metrics[metric_name]['steps'] = steps[sort_idx]
        all_metrics[metric_name]['values'] = values[sort_idx]
    
    # Plot each metric
    for metric_name, metric_data in all_metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(metric_data['steps'], metric_data['values'])
        plt.title(f"{metric_name} over Training Steps")
        plt.xlabel("Steps")
        plt.ylabel(metric_name)

        plt.savefig(os.path.join(fig_save_folder_path, "{}.png".format(metric_name)))
        plt.clf()
        
    return all_metrics

def list_available_metrics(folder_path):
    """List all available metrics in the event files."""
    all_metrics = set()
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith("events.out.tf"):
                event_file = os.path.join(root, file)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                all_metrics.update(ea.Tags()['scalars'])
    
    return sorted(list(all_metrics))

# Example usage
folder_path = "/N/slate/tnn3/TruongChu/merraRun/experiments/run_20250318_070053/lightning_logs/version_0" # Adjust 
# Example usage
available_metrics = list_available_metrics(folder_path)
print("Available metrics:", available_metrics)
plot_metrics_from_folder(folder_path, available_metrics)