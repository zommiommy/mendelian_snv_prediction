
from mendelian_snv_prediction import get_data, get_model


def run(path):
    train, test = get_data(
        path,
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
        epochs=100,
        verbose=1,
        use_multiprocessing=False,
        shuffle=True
    )


if __name__ == "__main__":
    run("./mendelian_snv.csv.gz")
