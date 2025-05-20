import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os

from evaluation.Accuracy import accuracy_func
from evaluation.F1_all import precision_all, recall_all, f1_all
from evaluation.ROC_AUC import roc_auc_func

device = 'cuda' if torch.cuda.is_available() else 'cpu'

evaluation_metrics = {
    "accuracy": accuracy_func(True),
    "precision_0": precision_all(0, True),
    "recall_0": recall_all(0, True),
    "f1_0": f1_all(0, True),
    "precision_1": precision_all(1, True),
    "recall_1": recall_all(1, True),
    "f1_1": f1_all(1, True),
    "precision_all": precision_all("weighted", True),
    "recall_all": recall_all("weighted", True),
    "f1_all": f1_all("weighted", True),
    "roc_auc": roc_auc_func
}

def evaluate_model(model, loader, folder_output, eval_name, threshold=0.5):
    model.eval()
    model.to(device)

    probs_total = np.empty((0, ))
    y_total = np.empty((0, ))
    filenames_total = []

    with torch.no_grad():
        for batch in loader:
            # x, y, file_name = batch
            x = batch["data"]
            y = batch["label"]
            file_name = batch["file_name"]
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = F.sigmoid(outputs).view(-1)

            probs_total = np.concatenate([probs_total, probs.cpu()])
            y_total = np.concatenate([y_total, y.cpu()])
            
            filenames_total.extend(file_name)

    # Save results to CSV
    results_df = pd.DataFrame({
        "filename": filenames_total,
        "label": y_total.astype(int),
        "probability": probs_total
    })
    results_df.to_csv(os.path.join(folder_output, "{}_results.csv".format(eval_name)), index=False)

    eval_results = {}

    for metric_name, metric_func in evaluation_metrics.items():
        eval_results[metric_name] = metric_func(probs_total, y_total, threshold)

    return eval_results
