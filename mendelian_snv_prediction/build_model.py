
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.metrics import AUC

from .utils import balanced_binary_crossentropy


def build_model(x, filters, kernels, pools, dense):
    # Convolutional part
    for _filter, _kernel, _pool in zip(filters, kernels, pools):
        x = Conv2D(_filter, _kernel, activation="relu", padding="same")(x)
        x = MaxPool2D(_pool)(x)

    x = Flatten()(x)

    # Dense part
    for _dense in dense:
        x = Dense(_dense, activation="relu")(x)

    x = Dense(1, activation="sigmoid")(x)
    return x


def get_model(windows_size=200, balancing_loss_weigth=10):
    filters = [128, 64, 64, 32, 32]
    kernels = [(9, 4), (6, 2), (3, 1), (3, 1), (3, 1)]
    pools = [(3, 1), (2, 1), (2, 2), (2, 2), (2, 1)]
    dense = [128, 64, 64, 32]

    i = Input(
        shape=(windows_size, 4)
    )
    x = Reshape((windows_size, 4, 1))(i)
    classifier = Model(
        inputs=i,
        outputs=build_model(
            x,
            filters,
            kernels,
            pools,
            dense
        )
    )

    classifier.compile(
        optimizer="nadam",
        loss=balanced_binary_crossentropy(balancing_loss_weigth),
        metrics=[
            "binary_accuracy",
            AUC(curve='PR')
        ]
    )
    return classifier