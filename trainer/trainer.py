import torch
import os

from lightning import Trainer
from lightning.pytorch.callbacks import Callback

from evaluation.total import evaluate_model

class CustomEarlyStopping(Callback):
    def __init__(self, patience=5, loss_threshold=0.1):
        super().__init__()
        self.patience = patience
        self.loss_threshold = loss_threshold
        self.best_loss = float('inf')
        self.counter = 0

    def on_validation_end(self, trainer, pl_module):
        # Retrieve validation loss
        val_loss = trainer.callback_metrics.get("val_loss")

        # If val_loss is None (not logged yet), skip
        if val_loss is None:
            return

        val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

        # Check if loss is under threshold
        if val_loss < self.loss_threshold:
            trainer.should_stop = True
            print(f"Stopping: validation loss {val_loss:.4f} < threshold {self.loss_threshold}")
            return

        # Check improvement
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"No improvement in val_loss for {self.counter} steps.")
            if self.counter >= self.patience:
                trainer.should_stop = True
                print(f"Early stopping after {self.patience} steps without improvement.")

# Custom Trainer that includes custom evaluation
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def custom_test(self, *args, **kwargs):
        print("🔍 Running custom evaluation inside Trainer...")
        model = kwargs.get("model", None)
        datamodule = kwargs.get("datamodule", None)
        folder_output = kwargs.get("folder_output", None)
        data_name = kwargs.get("data_name", None)

        assert model is not None, "⚠️ Custom evaluation skipped: No model provided."
        assert datamodule is not None, "⚠️ Custom evaluation skipped: No datamodule provided."
        assert folder_output is not None, "⚠️ Custom evaluation skipped: No file output path provided."
        assert data_name is not None, "⚠️ Custom evaluation skipped: No data module name provided."

        test_loader = datamodule.test_dataloader()
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
