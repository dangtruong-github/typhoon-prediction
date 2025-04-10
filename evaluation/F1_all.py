import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from evaluation.utils import prob_to_pred

def precision_all(label, prob=False):
    def precision_pred(preds, truth, threshold=0.5):
        if prob:
            preds = prob_to_pred(preds, threshold)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(truth, torch.Tensor):
            truth = truth.detach().cpu().numpy()

        # Calculate precision
        precision = precision_score(y_true=truth, y_pred=preds, 
                                    average='binary', pos_label=label,
                                    zero_division=0)

        return precision
    
    return precision_pred

def recall_all(label, prob=False):
    def recall_pred(preds, truth, threshold=0.5):
        if prob:
            preds = prob_to_pred(preds, threshold)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(truth, torch.Tensor):
            truth = truth.detach().cpu().numpy()
        # Calculate recall
        recall = recall_score(y_true=truth, y_pred=preds, 
                              average='binary', pos_label=label,
                              zero_division=0)


        return recall
    
    return recall_pred

def f1_all(label, prob=False):
    def f1_pred(preds, truth, threshold=0.5):
        if prob:
            preds = prob_to_pred(preds, threshold)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(truth, torch.Tensor):
            truth = truth.detach().cpu().numpy()
        # Calculate f1
        f1 = f1_score(y_true=truth, y_pred=preds, average='binary',
                      pos_label=label, zero_division=0)


        return f1
    
    return f1_pred
