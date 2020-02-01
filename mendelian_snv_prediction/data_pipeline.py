import numpy as np
import pandas as pd
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from ucsc_genomes_downloader.utils import wiggle_bed_regions


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
        train_bed = df.head(int(head_threshold))
    else:
        train_bed = df

    # Get all the positives
    positives = train_bed[train_bed.labels == 1]

    # WARNING
    # "Generate" more positives by offsetting the windows by a random value
    multiplied_positives = wiggle_bed_regions(positives, 150, 10, seed)

    # Generate the data for the assembly
    X = BedSequence(
        assembly=assembly,
        bed=train_bed,
        batch_size=batchsize
    )

    # Get the labels and covert them to float
    # becasue keras fit function break on int labels
    # with the custom loss
    y = train_bed.labels.values
    y = y.astype(float)

    # Merge the data into a sequence
    mixed = MixedSequence(X, y, 128)
    # Shuffle the data
    mixed.on_epoch_end()
    return mixed
