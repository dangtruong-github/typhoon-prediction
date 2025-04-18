# py
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from evaluation.total import evaluation_metrics


class BaseModel(L.LightningModule):
    def __init__(self, model, lr: float,
                 pos_weight: float, threshold: float=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.threshold = threshold
        self.pos_weight = torch.tensor([pos_weight])
        self.model = model

        self.__init_metrics()
        self.__init_log_history()
        
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def __init_metrics(self):
        # Metrics
        self.metrics = evaluation_metrics

    def __init_log_history(self):
        self.log_history = {
            "train_loss": [],
            "val_loss": [],
        }

        for metric_name in self.metrics.keys():
            self.log_history["train_{}".format(metric_name)] = []
            self.log_history["val_{}".format(metric_name)] = []

    def each_step(self, batch, type_step):
        # x, y, _ = batch
        x = batch["data"]
        y = batch["label"]
        logits = self.model.forward(x)

        loss = self.loss_fn(logits, y.float().view(-1, 1))
         # Convert logits to probabilities using sigmoid function
        probs = torch.sigmoid(logits).view(-1)

        print(probs)
        print(y)

        self.log('{}_loss'.format(type_step), loss,
                 prog_bar=(type_step=="train"))
        
        for metric_name, metric_func in self.metrics.items():
            metric_val = metric_func(probs, y, self.threshold)

            self.log("{}_{}".format(type_step, metric_name), metric_val,
                     prog_bar=False)
            
        return loss

    def training_step(self, batch, batch_idx):
        return self.each_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.each_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.each_step(batch, "test")

    def on_validation_epoch_end(self):
        for log_hist_type in self.log_history.keys():
            log_hist_val = self.trainer.callback_metrics.get(log_hist_type)
            if log_hist_val is not None:
                self.log_history[log_hist_type].append(log_hist_val.cpu().item())
            else:
                self.log_history[log_hist_type].append(None)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.model.load_state_dict(state_dict, *args, **kwargs)
