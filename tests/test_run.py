
import os
from mendelian_snv_prediction import get_data, get_model


def test_run():
    pwd = os.path.dirname(os.path.abspath(__file__))

    train, test = get_data(
        pwd + "/../mendelian_snv.csv.gz",
        assembly="hg19",
        batchsize=128,
        head_threshold=1e5,
        seed=1337
    )

    model = get_model()
    model.summary()

    model.fit_generator(
        generator=train,
        steps_per_epoch=train.steps_per_epoch,
        validation_data=test,
        validation_steps=test.steps_per_epoch,
        epochs=1,
        verbose=1,
        use_multiprocessing=False,
        shuffle=True
    )
