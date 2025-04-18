##########################
## main.py (Full version with: best model loading, timestamped checkpoints, smart cleanup)
##########################
import argparse
import yaml
import os
from datetime import datetime
import sys
import numpy as np

import torch

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info

from data.total import get_data_module
from models.total import get_model
from evaluation.total import evaluate_model
from evaluation.plotting import plot_all

# Custom Trainer that includes custom evaluation
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self, *args, **kwargs):
        print("🔍 Running custom evaluation inside Trainer...")
        model = self.model if hasattr(self, "model") else args[0]
        datamodule = kwargs.get("datamodule", None)
        folder_output = kwargs.get("folder_output", None)
        data_name = kwargs.get("data_name", None)

        assert datamodule is not None, "⚠️ Custom evaluation skipped: No datamodule provided."
        assert folder_output is not None, "⚠️ Custom evaluation skipped: No file output path provided."
        assert data_name is not None, "⚠️ Custom evaluation skipped: No data module name provided."

        test_loader = datamodule.val_dataloader()
        metrics_returned = evaluate_model(
            model.model, test_loader,
            folder_output, eval_name=data_name
        )

        file_output = os.path.join(folder_output, "evaluation.txt")

        if os.path.isfile(file_output) is False:
            f_output = open(file_output, "w")
            f_output.close()

        with open(file_output, "a") as f:
            f.write(f"{data_name}\n")
            for key, item in metrics_returned.items():
                print("{}: {:.5f}".format(key, item))
                f.write("{}: {:.5f}\n".format(key, item))

def cleanup_checkpoints(checkpoint_dir, keep_prefixes):
    for filename in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(full_path):
            # Check if the filename matches any keep_prefixes
            if any(filename.startswith(prefix) for prefix in keep_prefixes):
                continue  # Skip removing matching files
            os.remove(full_path)
            rank_zero_info(f"🧹 Removed old checkpoint: {filename}")

def parser_total():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_undersampling', type=float, default=-1)
    parser.add_argument('--pos_weight', type=float, default=-1)
    parser.add_argument('--pos_step', type=int, default=0)
    parser.add_argument('--config', type=str, default="configs/feature_expert_full.yaml")
    parser.add_argument('--model', type=str, default="N/A")
    args = parser.parse_args()

    # Load config file
    config = None
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

        print(f"Config:\n{config}")

    # Update arguments with values from the config file if they exist
    if config:
        for key, value in config.items():
            if hasattr(args, key) is False:
                setattr(args, key, value)
    else:
        raise ValueError(f"Invalid config file {args.config}")

    # Overwrite pos_weight if it's negative
    if args.pos_weight < 0:
        args.pos_weight = float(config.get('pos_weight', args.pos_weight))
    if args.rate_undersampling < 0:
        args.rate_undersampling = int(config.get('rate_undersampling', args.rate_undersampling))
    if args.model == "N/A":
        args.model = config.get('model', args.model)

    return args

def get_checkpoints():
    pass

def main():
    args = parser_total()

    print(args)
    # Timestamped unified experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join("experiments", args.model,
                                  f"pos_{args.pos_step}",
                                  "rus_{:.2f}_pos_{:.2f}".format(args.rate_undersampling, args.pos_weight),
                                  f"run_{timestamp}")
    
    setattr(args, "exp_dir", experiment_dir)

    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    sys.stdout = open(os.path.join(experiment_dir, "out.txt"), "w")
    sys.stderr = open(os.path.join(experiment_dir, "err.txt"), "w")

    # Save the dictionary to a YAML file
    args_dict = vars(args)
    with open(os.path.join(experiment_dir, "configs.yaml"), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)

    # Data
    data_module = get_data_module(args)

    print(len(data_module.train_set))
    print(len(data_module.val_set))
    print(len(data_module.test_set))
    print(len(data_module.train_dataloader()))
    print(len(data_module.val_dataloader()))
    print(len(data_module.test_dataloader()))
    
    print(data_module.train_set.df["Label"].value_counts())
    print(data_module.val_set.df["Label"].value_counts())
    print(data_module.test_set.df["Label"].value_counts())

    # Model
    model = get_model(args)

    # TensorBoard Logger
    tb_logger = TensorBoardLogger(save_dir=experiment_dir, name="lightning_logs")

    # Checkpoint Callbacks
    best_val_loss_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val_loss',
        mode='min',  # Use 'min' since lower loss is better
        save_top_k=3,
        filename='best-val_loss-{epoch:02d}-{val_loss:.2f}',
    )

    # Checkpoint for best val_f1
    best_val_f1_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val_f1_1',
        mode='max',  # Use 'max' since higher F1 score is better
        save_top_k=3,
        filename='best-val_f1-{epoch:02d}-{val_f1:.2f}',
    )

    periodic_ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_epochs=args.save_every_n_epochs,
        save_top_k=-1,
        filename='epoch-{epoch:02d}',
    )

    # Trainer
    trainer = CustomTrainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        callbacks=[best_val_loss_cb, best_val_f1_cb, periodic_ckpt_cb],
        logger=tb_logger,
        log_every_n_steps=args.log_every_n_steps
    )

    # Train
    trainer.fit(model, datamodule=data_module)

    # Plot validation loss
    plot_all(model.log_history, experiment_dir)

    # Load best model for testing
    if best_val_loss_cb.best_model_path:
        print(f"🏆 Loading best model from {best_val_loss_cb.best_model_path}")

        checkpoint = torch.load(best_val_loss_cb.best_model_path, weights_only=False)

        print(checkpoint.keys())

        # If the checkpoint contains a model state dictionary, load it into your model
        model.model.load_state_dict(checkpoint['state_dict'])

        # Test with best model
        trainer.test(model, datamodule=data_module,
                     data_name="val_loss", folder_output=args.exp_dir)

    # Load best model for testing
    if best_val_f1_cb.best_model_path:
        print(f"🏆 Loading best model from {best_val_f1_cb.best_model_path}")
        checkpoint = torch.load(best_val_f1_cb.best_model_path, weights_only=False)

        print(checkpoint.keys())

        # If the checkpoint contains a model state dictionary, load it into your model
        model.model.load_state_dict(checkpoint['state_dict'])
        # Test with best model
        trainer.test(model, datamodule=data_module,
                     data_name="val_f1", folder_output=args.exp_dir)

    # Clean up old checkpoints except best
    # Example usage:
    cleanup_checkpoints(
        checkpoint_dir=ckpt_dir,
        keep_prefixes=["best-val_loss", "best-val_f1"]
    )

    sys.stdout.close()
    sys.stderr.close()

if __name__ == '__main__':
    main()
