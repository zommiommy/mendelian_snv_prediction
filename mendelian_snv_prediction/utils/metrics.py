from tensorflow.keras.metrics import AUC


def get_model_metrics():
    return [
        "binary_accuracy",
        AUC(curve='PR', name="auprc"),
        AUC(curve='ROC', name="auroc"),
    ]
