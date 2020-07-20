from mendelian_snv_prediction import get_holdouts, get_model


def test_run():
    """Test that the complete pipeline works."""
    window_size = 50
    for train, test in get_holdouts(
        holdouts=1,
        max_wiggle_size=10,
        batch_size=16,
        window_size=window_size,
        nrows=1000,
        verbose=False
    ):

        model = get_model(window_size=window_size)

        model.fit(
            train,
            steps_per_epoch=train.steps_per_epoch,
            validation_data=test,
            validation_steps=test.steps_per_epoch,
            verbose=False
        )
