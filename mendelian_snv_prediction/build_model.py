
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

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


def get_model():
    filters = [128, 64, 64, 32]
    kernels = [(9, 4), (6, 2), (3, 1), (3, 1)]
    pools = [(3, 1), (2, 1), (2, 2), (2, 2)]
    dense = [256, 64, 32]

    i = Input(
        shape=(500, 4)
    )
    x = Reshape((500, 4, 1))(i)
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
        loss=balanced_binary_crossentropy(2.0)
    )
    return classifier