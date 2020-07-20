import pandas as pd
import os
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence, VectorSequence
from ucsc_genomes_downloader.utils import wiggle_bed_regions, expand_bed_regions
from ucsc_genomes_downloader import Genome
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm


def create_sequence(bed: pd.DataFrame, assembly: Genome, batch_size: int) -> MixedSequence:
    """Return training sequence.

    Parameters
    ----------------------------
    bed: pd.DataFrame,
        Dataframe with bed file structure.
    assembly: Genome,
        Genomic assembly to use.
    batch_size: int,
        Batch size to use.

    Returns
    ----------------------------
    Training sequence for model.
    """
    return MixedSequence(
        x=BedSequence(assembly=assembly, bed=bed, batch_size=batch_size),
        y=VectorSequence(bed.labels.values, batch_size=batch_size)
    )


def get_holdouts(
    batch_size: int = 128,
    max_wiggle_size: int = 150,
    wiggles: int = 10,
    random_state: int = 42,
    holdouts: int = 10,
    window_size: int = 500,
    test_size: float = 0.3,
    verbose: bool = True,
    nrows: int = None
):
    """Return generator with training and testing holdouts.

    Parameters
    ---------------------------
    batch_size: int = 128,
        The batch size to use.
        Since the task is significantly unbalances, consider using high
        batch sizes.
    max_wiggle_size: int = 150,
        Amount to wiggle the windows.
    wiggles: int = 10,
        Number of wiggles per sample.
    random_state: int = 42,
        Random state to use for reproducibility.
    holdouts: int = 10,
        Number of holdouts to train over.
    window_size: int = 500,
        Window size to use.
    test_size: float = 0.3,
        Percentage to leave for the test set.
    verbose: bool = True
        Wethever to show or not the loading bar.
    nrows: int = None,
        Number of rows to read. Useful to test the pipeline.

    Raises
    ----------------------------
    ValueError,
        If given window size if less or equal than the double of given
        maximum wiggle size.

    Returns
    ----------------------------
    Generator with the training holdouts.
    """
    if window_size <= max_wiggle_size*2:
        raise ValueError(
            (
                "Given window size {} is less or equal than twice the "
                "given max_wiggle_size {}. This may lead the central SNV "
                "to fall outside the region, hence causing a false positive. "
                "Please either increase the window size or reduce the "
                "maximum wiggle size."
            ).format(
                window_size,
                max_wiggle_size
            )
        )

    # Load the bed file
    bed = pd.read_csv(
        "{}/mendelian_snv.csv.gz".format(
            os.path.dirname(os.path.abspath(__file__))
        ),
        nrows=nrows
    )

    # Expand (or compress) given bed file windows to required size
    bed = expand_bed_regions(bed, window_size)

    # Load the genomic assembly
    assembly = Genome("hg19", verbose=False)

    # Prepare the object that generates the holdout indices
    holdouts_generator = StratifiedShuffleSplit(
        n_splits=holdouts,
        test_size=test_size,
        random_state=random_state
    )

    # For each holdout
    for i, (train, test) in tqdm(
        enumerate(holdouts_generator.split(bed, bed.labels)),
        desc="Holdouts",
        disable=not verbose,
        total=holdouts
    ):
        # We get the training bed partition
        train_bed = bed.iloc[train]
        # And the testing bed partition
        test_bed = bed.iloc[test]
        # We wiggle the bed regions the desired amount to generate
        # the required amount of wiggles.
        # We wiggle only the training positives, as wiggling the training
        # negatives might create false negatives.
        positives = train_bed[(train_bed.labels == 1).values]
        # Computing the wiggles
        wiggled_train_bed = wiggle_bed_regions(
            positives,
            max_wiggle_size,
            wiggles,
            random_state
        )
        # Concatenig the training data
        train_bed = pd.concat([
            wiggled_train_bed,
            train_bed
        ])
        # Shuffle the training data
        train_bed = train_bed.sample(frac=1, random_state=random_state+i)
        # And we return the computed training sequences.
        yield (
            create_sequence(train_bed, assembly, batch_size),
            create_sequence(test_bed, assembly, batch_size)
        )
