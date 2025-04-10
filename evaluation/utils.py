def prob_to_pred(probs, threshold=0.5):
    preds = (probs >= threshold)

    return preds
