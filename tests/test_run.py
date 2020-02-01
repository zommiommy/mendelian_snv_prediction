
import os
from mendelian_snv_prediction import get_data, get_model


def test_run():
    pwd = os.path.dirname(os.path.abspath(__file__))
    mixed = get_data(pwd + "/../mendelian_snv.csv.gz")

    classifier = get_model()
    classifier.summary()

    histoty = classifier.fit_generator(
        generator=mixed,
        steps_per_epoch=mixed.steps_per_epoch // 5,
        epochs=2,
        verbose=1,
        use_multiprocessing=False,
        shuffle=True
    ).history
