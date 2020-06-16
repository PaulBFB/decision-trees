import numpy as np
from math import log2


def entropy(data: np.array) -> float:
    """
    calculates the entropy of an outcome array of 1 and 0

    :param data: a np.array of outcomes
    :return: float
    """

    assert data.ndim == 1, f"input data must be 1 dimensional array - input has {data.ndim}"

    # for classes of 1 and 0 get probabilities
    probability_ones = data.sum() / len(data)
    probability_zeros = 1 - probability_ones
    
    if any((probability_ones == 0, probability_zeros == 0)):
        return 0

    # standard entropy formula
    e = -(probability_ones * log2(probability_ones) + probability_zeros * log2(probability_zeros))

    return e


def information_gain(before_split: np.array,
                     splits: list) -> float:
    """
    takes an original array and two sub-array and calculates information gain in bits based on entropy base 2

    :param before_split: array, 1 and 0
    :param splits: list of arrays, sub-arrays of before_splits
    :return: float
    """

    assert len(before_split) == sum([len(split) for split in splits]), f"splits must add up to length of original{len(original)}/{sum([len(i) for i in splits])}"
    assert sum(before_split) == sum([sum(split) for split in splits]), "probabilities of sub arrays do not sum to same total as original"

    # get parameters all arrays - size, entropy
    # only for convenience / readability
    size_before = len(before_split)
    entropy_before = entropy(before_split)

    entropy_after = 0

    # add all the partial entropies of the sub-parts
    for split in splits:
        # get the size of the part and it's entropy
        split_size = len(split)
        entropy_split = entropy(split)

        # get the entropy contribution of the part, add it to entropy_before
        relative_size = split_size / size_before
        entropy_contribution = entropy_split * relative_size

        entropy_after += entropy_contribution

    ig = entropy_before - entropy_after

    return ig


if __name__ == '__main__':
    test_data = np.array([1, 0, 0, 0, 0, 0])
    test_data_ig_before = np.array([1] * 13 + [0] * 7)

    test_data_ig_after_p1 = np.array([1] * 7 + [0])
    test_data_ig_after_p2 = np.array([1] * 6 + [0] * 6)

    ig_test = information_gain(test_data_ig_before,
                               [test_data_ig_after_p1, test_data_ig_after_p2])
    print(ig_test)
