import pandas as pd
import os
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from ucsc_genomes_downloader.utils import wiggle_bed_regions
from ucsc_genomes_downloader import Genome
from sklearn.model_selection import StratifiedShuffleSplit


def create_sequence(bed, assembly, batchsize):
    return MixedSequence(
        BedSequence(
            assembly=assembly,
            bed=bed,
            batch_size=batchsize
        ),
        bed.labels,
        batchsize
    )


def get_data(
    batchsize: int = 128,
    head_threshold: int = 1e5,
    max_wiggle_size=150,
    wiggles=10,
    seed=42
):
    # Load the bed file
    df = pd.read_csv(
        "{}/mendelian_snv.csv.gz".format(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    train, test = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.3
        ).split(df, df.labels))
        
    train_df = df.iloc[train]
    test_df = df.iloc[test]
    train_df = wiggle_bed_regions(train_df, max_wiggle_size, wiggles, seed)

    assembly = Genome("hg19")

    return (
        create_sequence(train_df, assembly, batchsize),
        create_sequence(test_df, assembly, batchsize)
    )
