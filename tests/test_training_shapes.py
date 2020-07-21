from mendelian_snv_prediction import get_holdouts


def test_training_shape():
    batch_size = 16
    window_size = 50
    train, test = next(get_holdouts(
        batch_size=batch_size,
        window_size=window_size,
        max_wiggle_size=10,
        nrows=1000,
        verbose=False
    ))
    assert train[0][0].shape == (
        batch_size,
        window_size,
        4
    )
    assert test[0][0].shape == (
        batch_size,
        window_size,
        4
    )
