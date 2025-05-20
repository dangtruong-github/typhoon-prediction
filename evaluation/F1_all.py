import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from evaluation.utils import prob_to_pred

def precision_all(label_type, prob=False):
    def precision_pred(preds, truth, threshold=0.5):
        if prob:
            preds = prob_to_pred(preds, threshold)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(truth, torch.Tensor):
            truth = truth.detach().cpu().numpy()

        # Calculate precision
        if label_type != "weighted":
            precision = precision_score(y_true=truth, y_pred=preds, 
                                        average='binary', pos_label=label_type,
                                        zero_division=0)
        else:
            precision = precision_score(y_true=truth, y_pred=preds, 
                                        average="weighted", zero_division=0)


        return precision
    
    return precision_pred

def recall_all(label_type, prob=False):
    def recall_pred(preds, truth, threshold=0.5):
        if prob:
            preds = prob_to_pred(preds, threshold)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(truth, torch.Tensor):
            truth = truth.detach().cpu().numpy()

        # Calculate recall
        if label_type != "weighted":
            recall = recall_score(y_true=truth, y_pred=preds, 
                                        average='binary', pos_label=label_type,
                                        zero_division=0)
        else:
            recall = recall_score(y_true=truth, y_pred=preds, 
                                        average="weighted", zero_division=0)


        return recall
    
    return recall_pred

def f1_all(label_type, prob=False):
    def f1_pred(preds, truth, threshold=0.5):
        if prob:
            preds = prob_to_pred(preds, threshold)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(truth, torch.Tensor):
            truth = truth.detach().cpu().numpy()

        # Calculate f1
        if label_type != "weighted":
            f1 = f1_score(y_true=truth, y_pred=preds, 
                                        average='binary', pos_label=label_type,
                                        zero_division=0)
        else:
            f1 = f1_score(y_true=truth, y_pred=preds, 
                                        average="weighted", zero_division=0)

        return f1
    
    return f1_pred
