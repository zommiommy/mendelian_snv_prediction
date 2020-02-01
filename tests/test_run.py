
import os
from mendelian_snv_prediction import get_data, get_model
from holdouts_generator import holdouts_generator, balanced_random_holdouts

def test_run():
    pwd = os.path.dirname(os.path.abspath(__file__))

    mixed = get_data(
        pwd + "/../mendelian_snv.csv.gz",
        assembly="hg19",
        batchsize=128,
        head_threshold=1e5,
        seed=1337
    )


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
