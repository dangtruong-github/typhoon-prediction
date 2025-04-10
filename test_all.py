import os
import argparse
import yaml
import sys

from test import test_folder

def parser_test(folder):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    setattr(args, "folder", folder)

    
    # Load config from experiment folder
    config_path = os.path.join(args.folder, "configs.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print(f"Config:\n{config}")

    # Update arguments with values from the config file if they exist
    if config:
        for key, value in config.items():
            if hasattr(args, key) is False:
                setattr(args, key, value)
    else:
        raise ValueError(f"Invalid config file {args.config}")

    try:
        print(args.model)
    except:
        setattr(args, "model", args.folder.split("/")[-3])
        print(args.model)

    return args

def get_all_run_folders(directory):
    run_folders = []
    # Traverse through all subdirectories
    for root, dirs, _ in os.walk(directory):
        for folder in dirs:
            if folder.startswith("run_"):
                run_folders.append(os.path.join(root, folder))
    return run_folders

def main():
    # Example usage
    directory_path = "./experiments"  # Replace with the top-level folder path
    run_folders = get_all_run_folders(directory_path)
    print("Folders with 'run_' prefix across all levels:")
    for folder in run_folders:
        print(folder)

        args = parser_test(folder)

        test_folder(args)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()
