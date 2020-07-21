
from keras_mixed_sequence import Sequence
from tensorflow.keras.models import Model


def evaluate_sequence(model: Model, sequence: Sequence, verbose: bool):
    """Return dictionary with model evaluation.

    Parameters
    --------------------------------
    model: Model,
        The model to evaluate.
    sequence: Sequence,
        The sequences over which we evaluate the model.
    verbose: bool,
        Wethever to show or not the loading bar.

    Returns
    --------------------------------
    Dictionary with results.
    """
    return {
        "model": model.name,
        **dict(zip(
            model.metrics_names,
            model.evaluate_generator(
                sequence,
                sequence.steps_per_epoch,
                verbose=verbose
            ),
        ))
    }


def evaluate_model(model: Model, train: Sequence, test: Sequence, verbose: bool = True):
    """Evaluate the model on training and testing sequences.

    Parameters
    -------------------------------
    model: Model,
        The model to evaluate.
    train: Sequence,
        The training sequence to evaluate.
    test: Sequence,
        The testing sequence to evaluate.
    verbose: bool = True,
        Wethever to show or not the loading bar.
        By default this is True.

    Returns
    --------------------------------
    Dictionary with results.
    """
    return [
        {"run": "train", **evaluate_sequence(model, train, verbose)},
        {"run": "test", **evaluate_sequence(model, test, verbose)}
    ]
