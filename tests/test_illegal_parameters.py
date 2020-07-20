import pytest
from mendelian_snv_prediction import get_holdouts


def test_illegal_parameters():
    """Test that the proper exceptions are raised."""
    with pytest.raises(ValueError):
        next(get_holdouts(
            max_wiggle_size=150,
            window_size=200
        ))