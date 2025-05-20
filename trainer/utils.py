import os

from lightning.pytorch.utilities import rank_zero_info

def cleanup_checkpoints(checkpoint_dir, keep_prefixes):
    for filename in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(full_path):
            # Check if the filename matches any keep_prefixes
            if any(filename.startswith(prefix) for prefix in keep_prefixes):
                continue  # Skip removing matching files
            os.remove(full_path)
            rank_zero_info(f"🧹 Removed old checkpoint: {filename}")