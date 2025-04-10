import torch

from sklearn.metrics import roc_auc_score

def roc_auc_func(probs_total, y_total, threshold=0.5):
    if isinstance(probs_total, torch.Tensor):
        probs_total = probs_total.detach().cpu().numpy()

    if isinstance(y_total, torch.Tensor):
        y_total = y_total.detach().cpu().numpy()
    
    return roc_auc_score(y_true=y_total, y_score=probs_total)
