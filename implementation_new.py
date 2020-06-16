import pandas as pd
import numpy as np
from math import log2


t_col = 'Survived'


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


def load_data(path: str) -> pd.DataFrame:
    """
    load string wrapper - pandas will infer data types for us here
    :param path:
    :return:
    """
    with open(path, mode='r') as file:
        data = pd.read_csv(file, dtype={'Survived': bool})

    return data


def split_numeric(attribute: str,
                  value: float,
                  data: pd.DataFrame) -> tuple:
    """
    right_heavy split of numeric attributre  left -> < value, right >= value

    :param attribute:
    :param value:
    :param data:
    :param target_column:
    :return:
    """

    assert np.issubdtype(data[attribute].dtype, np.number), 'selected column must be numeric'
    assert not any(df[attribute].isna()), 'split column cannot contain blank values, please fill before using split'

    split_below = data.loc[data[attribute] < value]
    split_above = data.loc[data[attribute] >= value]

    return split_above, split_below


def split_non_numeric(attribute: str,
                      data: pd.DataFrame) -> tuple:

    assert not np.issubdtype(data[attribute].dtype, np.number), 'selected can\'t be numeric'
    assert not any(df[attribute].isna()), 'split column cannot contain blank values, please fill before using split'

    outcomes = data[attribute].unique()
#    print(len(outcomes))

    return tuple(data.loc[data[attribute] == outcome] for outcome in outcomes)


def information_gain(before_split: np.array,
                     splits: tuple,
                     target_column: str = t_col) -> float:
    """
    takes an original array and multiple sub-arrays and calculates information gain in bits based on entropy base 2

    :param before_split: array, 1 and 0
    :param splits: list of arrays, sub-arrays of before_splits
    :return: float
    """

    assert len(before_split) == sum([len(split) for split in splits]), f"splits must add up to length of original{len(before_split)}/{sum([len(i) for i in splits])}"

    # get parameters all arrays - size, entropy
    # only for convenience / readability
    size_before = len(before_split)
    entropy_before = entropy(before_split[target_column])

    entropy_after = 0

    # add all the partial entropies of the sub-parts
    for split in splits:
        # get the size of the part and it's entropy
        split_size = len(split)
        entropy_split = entropy(split[target_column])

        # get the entropy contribution of the part, add it to entropy_before
        relative_size = split_size / size_before
        entropy_contribution = entropy_split * relative_size

        entropy_after += entropy_contribution

    ig = entropy_before - entropy_after

    return ig


def find_ideal_split(data: pd.DataFrame,
                     criterion: str = 'information_gain') -> dict:
    """
    look over all attributes of the data, numeric and non-numeric, get best splits for them
    :param data:
    :param criterion
    :return:
    """

    attributes = filter(lambda x: x != t_col, data.columns)

    if criterion == 'gini':
        pass

    split_attribute = None
    best_ig = 0
    split_value = 0
    best_splits = None

    for attribute in attributes:
        if np.issubdtype(data[attribute].dtype, np.number):
            print(f'{attribute} - numeric')

            for i in data[attribute].unique():

                parts = split_numeric(attribute=attribute, value=i, data=data)
                ig = information_gain(before_split=data, splits=parts)

                if ig > best_ig:

                    best_ig = ig
                    split_attribute = attribute
                    split_value = i
                    best_splits = parts

        else:
            print(f'{attribute} - non numeric')

            parts = split_non_numeric(attribute=attribute, data=data)
#            print(parts)
            ig = information_gain(before_split=data, splits=parts)

            if ig > best_ig:
                best_ig = ig
                split_value = None
                split_attribute = attribute
                best_splits = parts

    result = {'value': split_value,
              'split_attribute': split_attribute,
              f'best_{criterion}': best_ig,
              'splits': best_splits}

    return result


def gini_index(before_split: pd.DataFrame,
               splits: tuple,
               outcomes: list,
               target_column: str = t_col) -> float:
    # get total elements to calculate shares
    total_elements = before_split.shape[0]

    gini = 0

    for split in splits:
        split_size = len(split) / total_elements

        if split_size == 0:
            continue

        score = 0

        for outcome in outcomes:
            # get fraction of cases where target == outcome
            p = np.mean(split[split[target_column] == outcome])
            score += p ** 2

        gini += (1.0 - score) * (split_size / total_elements)

    return gini


if __name__ == '__main__':
    df = load_data('./data/titanic.csv')
#    print(df.dtypes)

#    df = df.apply(lambda x: x.fillna(x.median()) if np.issubdtype(x.dtype, np.number), axis=0)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df.dropna(subset=['Embarked'], inplace=True)
    df.drop(columns=['Ticket', 'Name', 'Cabin', 'PassengerId'], inplace=True)

    print(df.isna().mean().sort_values(ascending=False))

#    print(information_gain(df, split_numeric('Age', value=25, data=df)))
#    print(gini_index(df, split_numeric('Age', value=25, data=df), [False, True]))

#    find_ideal_split(df)
#    for i in df.columns:
        #if df[i].dtype in (np.int64, np.float64):
         #   continue
        #print(i)
        #split_non_numeric(i, df)

    best_initial = find_ideal_split(df)
    print(best_initial)
