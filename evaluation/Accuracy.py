from evaluation.utils import prob_to_pred

def accuracy_func(prob=False):
    def accuracy_pred(preds, truth, threshold=0.5):
        if prob:
            preds = prob_to_pred(preds, threshold)
        # Calculate the number of correct predictions
        correct = (preds == truth).sum()

        # Calculate the accuracy
        accuracy = correct / len(truth)

        return accuracy

    return accuracy_pred
