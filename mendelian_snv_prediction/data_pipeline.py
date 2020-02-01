import numpy as np
import pandas as pd
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from ucsc_genomes_downloader.utils import wiggle_bed_regions
from holdouts_generator import holdouts_generator, balanced_random_holdouts

def create_sequence(bed, y, assembly="hg19", batchsize=128):
    X = BedSequence(
        assembly=assembly,
        bed=bed,
        batch_size=batchsize
    )
    return MixedSequence(X, y, batchsize)

def wiggle(training):
    x, y = training
    positives = x[y==1]
    x = pd.concat([x, wiggle_bed_regions(positives, 150, 10, seed=42)], axis=0)
    y = x.labels.values
    return create_sequence(x, y)

def split_train_test(bed, split_ratio=0.3):
    generator = holdouts_generator(
        bed, bed.labels,
        holdouts=balanced_random_holdouts(
            [split_ratio],
            [1]
        )
    )
    return next(generator())[0]

def get_data(
    path: str,
    assembly: str = "hg19",
    batchsize: int = 128,
    head_threshold: int = 1e5,
    seed : int = 1337
):
    # Load the bed fil
    df = pd.read_csv(path)

    # take a subsection of the dataset
    if head_threshold:
        bed = df.head(int(head_threshold))
    else:
        bed = df

    training, testing = split_train_test(bed)
    train = wiggle(training)
    test = create_sequence(*testing)

    return train, test