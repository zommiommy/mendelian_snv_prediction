from mendelian_snv_prediction import get_holdouts, build_deep_enhancers
from mendelian_snv_prediction.utils import evaluate_model
import shutil
import pandas as pd


def test_run():
    """Test that the complete pipeline works."""
    window_size = 30

    train, test = next(get_holdouts(
        max_wiggle_size=2,
        batch_size=16,
        window_size=window_size,
        nrows=10000,
        verbose=True
    ))

    model = build_deep_enhancers(window_size=window_size)

    model.fit(
        train,
        steps_per_epoch=train.steps_per_epoch,
        validation_data=test,
        validation_steps=test.steps_per_epoch,
        verbose=True
    )

    evaluate_model(model, train, test, True)
