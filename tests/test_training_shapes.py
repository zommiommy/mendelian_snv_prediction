from mendelian_snv_prediction import get_holdouts


def test_training_shape():
    batch_size = 128
    for window_size in (400, 1000):
        train, test = next(get_holdouts(
            batch_size=batch_size,
            window_size=window_size,
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
