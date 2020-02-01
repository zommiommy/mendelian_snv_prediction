
from mendelian_snv_prediction import get_data, get_model
import os

def run(path):
    mixed = get_data(path)

    classifier = get_model()
    classifier.summary()

    histoty = classifier.fit_generator(
        generator=mixed,
        steps_per_epoch=mixed.steps_per_epoch // 5,
        epochs=100,
        verbose=1,
        use_multiprocessing=False,
        shuffle=True
    ).history


if __name__ == "__main__":
    run("./mendelian_snv.csv.gz")
