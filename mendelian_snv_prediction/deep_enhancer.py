from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Input, Conv1D, MaxPool1D, Activation
from tensorflow.keras.metrics import AUC
from .utils import get_model_metrics


def build_deep_enhancers(window_size: int, nucleotides_number: int = 4) -> Sequential:
    """Return Deep Enhancers fixed model.

    Parameters
    --------------------------
    window_size: int,
        Window size of the nucleotides windows.
    nucleotides_number: int,
        Number of nucleotides considered in each window.
        By default, the value is 4.

    Returns
    --------------------------
    DeepEnhancer model.

    References
    --------------------------
    https://www.nature.com/articles/nmeth.2987
    """
    model = Sequential([
        Input((window_size, nucleotides_number)),
        Conv1D(filters=128, kernel_size=8),
        BatchNormalization(),
        Activation("relu"),
        Conv1D(filters=128, kernel_size=8),
        BatchNormalization(),
        Activation("relu"),
        MaxPool1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3),
        BatchNormalization(),
        Activation("relu"),
        Conv1D(filters=64, kernel_size=3),
        BatchNormalization(),
        Activation("relu"),
        MaxPool1D(pool_size=2),
        Flatten(),
        Dense(units=256, activation="relu"),
        Dropout(rate=0.1),
        Dense(units=128, activation="relu"),
        Dropout(rate=0.1),
        Dense(units=1, activation="sigmoid"),
    ], name="DeepEnhancer")

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=get_model_metrics()
    )

    return model
