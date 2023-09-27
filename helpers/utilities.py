import itertools
from typing import List


def flatten_chromosomes(chromosomes) -> List:
    """
    Flatten a list of lists or handle other types of data gracefully.

    Args:
        chromosomes (list or any): The data to be flattened or handled.

    Returns:
        list: The flattened list if input is a list of lists,
              the input list as-is if it's a regular list,
              an empty list for other types of data.
    """
    if isinstance(chromosomes, list) and all(isinstance(item, list) for item in chromosomes):
        # Flatten a list of lists
        return list(itertools.chain.from_iterable(chromosomes))
    elif isinstance(chromosomes, list):
        # Handle the case when it's just a regular list, not a list of lists
        return chromosomes
    else:
        # Handle other types of data as needed
        return []
