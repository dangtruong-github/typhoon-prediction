import argparse
import os
import yaml
import sys
import torch

from data.total import get_data_module
from models.total import get_model
from trainer.trainer import CustomTrainer

def parser_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help="Path to the experiment folder")
    args = parser.parse_args()

    
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

def test_folder(args):
    sys.stdout = open(os.path.join(args.folder, "out_test.txt"), "w")
    sys.stderr = open(os.path.join(args.folder, "err_test.txt"), "w")
    
    # Data
    data_module = get_data_module(args)
    
    # Model
    model = get_model(args)
    
    # Load best checkpoint
    ckpt_dir = os.path.join(args.folder, "checkpoints")
    
    # Test
    trainer = CustomTrainer()

    for item in os.listdir(ckpt_dir):
        if item[-5:] != ".ckpt":
            continue
        best_ckpt_path = os.path.join(ckpt_dir, item)
        print(f"🏆 Loading best model from {best_ckpt_path}")

        checkpoint = torch.load(best_ckpt_path, weights_only=False)

        # If the checkpoint contains a model state dictionary, load it into your model
        model.model.load_state_dict(checkpoint['state_dict'])

        trainer.custom_test(model=model, datamodule=data_module,
                     data_name=item, folder_output=ckpt_dir)

    sys.stdout.close()
    sys.stderr.close()

def main():
    args = parser_test()

    test_folder(args)
    
    
if __name__ == '__main__':
    main()
