import numpy as np
from math import log2


def entropy(data: np.array):
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
                     split_part_1: np.array,
                     split_part_2: np.array):
    """
    takes an original array and two sub-array and calculates information gain in bits based on entropy base 2

    :param before_split: array, 1 and 0
    :param split_part_1: sub-array of original, 1 and 0
    :param split_part_2: sub-array of original, 1 and 0
    :return: float
    """

    assert len(before_split) == (len(split_part_1) + len(split_part_2)), f"splits must add up to length of original{len(original)}/{len(split_part_1)+len(split_part_2)}"
    assert sum(before_split) == sum(split_part_1) + sum(split_part_2), "probabilities of sub arrays do sum to same total as original"

    # get parameters all arrays - size, entropy
    # only for convenience / readability
    size_before = len(before_split)
    entropy_before = entropy(before_split)

    size_part_1 = len(split_part_1)
    entropy_part_1 = entropy(split_part_1)

    size_part_2 = len(split_part_2)
    entropy_part_2 = entropy(split_part_2)

    # after splitting entropy calculated in weights
    # the sub-arrays each contribute entropy in relation to their size
    # thereby, the largest overall purity increase will be achieved
    entropy_share_p1 = (size_part_1 / size_before) * entropy_part_1
    entropy_share_p2 = (size_part_2 / size_before) * entropy_part_2

    ig = entropy_before - (entropy_share_p1 + entropy_share_p2)

    return ig


if __name__ == '__main__':
    test_data = np.array([1, 0, 0, 0, 0, 0])
    test_data_ig_before = np.array([1] * 13 + [0] * 7)

    test_data_ig_after_p1 = np.array([1] * 7 + [0])
    test_data_ig_after_p2 = np.array([1] * 6 + [0] * 6)

    ig_test = information_gain(test_data_ig_before, test_data_ig_after_p1, test_data_ig_after_p2)
    print(ig_test)
