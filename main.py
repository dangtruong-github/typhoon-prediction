##########################
## main.py (Full version with: best model loading, timestamped checkpoints, smart cleanup)
##########################
import yaml
import os
from datetime import datetime
import sys
import numpy as np

import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from data.total import get_data_module
from models.total import get_model
from evaluation.plotting import plot_all
from configs.parser import parser_total
from trainer.trainer import CustomTrainer, CustomEarlyStopping
from trainer.utils import cleanup_checkpoints

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

    dummy_data = data_module.train_set.__getitem__(0)
    print(f"dummy_data['data'].shape: {dummy_data["data"].shape}")
    print(data_module.train_set.type_retrieve)

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
        filename='best-val_f1-{epoch:02d}-{val_f1_1:.2f}',
    )

    periodic_ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_epochs=args.save_every_n_epochs,
        save_top_k=-1,
        filename='epoch-{epoch:02d}',
    )

    early_stopping_cb = CustomEarlyStopping(patience=5, loss_threshold=1e-4)

    # Trainer
    trainer = CustomTrainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        callbacks=[best_val_loss_cb, best_val_f1_cb, periodic_ckpt_cb, early_stopping_cb],
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
        trainer.custom_test(model=model, datamodule=data_module,
                     data_name="val_loss", folder_output=args.exp_dir)

    # Load best model for testing
    if best_val_f1_cb.best_model_path:
        print(f"🏆 Loading best model from {best_val_f1_cb.best_model_path}")
        checkpoint = torch.load(best_val_f1_cb.best_model_path, weights_only=False)

        print(checkpoint.keys())

        # If the checkpoint contains a model state dictionary, load it into your model
        model.model.load_state_dict(checkpoint['state_dict'])
        # Test with best model
        trainer.custom_test(model=model, datamodule=data_module,
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
