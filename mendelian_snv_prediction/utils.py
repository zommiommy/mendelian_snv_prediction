import tensorflow as tf
import tensorflow.keras as k


def balanced_binary_crossentropy(true_weight=1.0):
    def loss(y_true, y_pred):
        mask = y_true == 1
        nmask = ~mask
        l1 = tf.cond(
            k.backend.any(mask),
            true_fn=lambda: k.backend.categorical_crossentropy(
                y_true[mask], y_pred[mask]),
            false_fn=lambda: 0.0,
            name=None
        )
        l0 = tf.cond(
            k.backend.any(nmask),
            true_fn=lambda: k.backend.categorical_crossentropy(
                y_true[nmask], y_pred[nmask]),
            false_fn=lambda: 0.0,
            name=None
        )

        return l0 + true_weight * l1
    return loss
