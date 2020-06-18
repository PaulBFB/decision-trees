import pandas as pd
import numpy as np
from math import log2
from pprint import pprint


t_col = 'Survived'
p_case = 1


def entropy(data: np.array) -> float:
    """
    calculates the entropy of an outcome array of 1 and 0

    :param data: a np.array of outcomes
    :return: float
    """

    assert data.ndim == 1, f"input data must be 1 dimensional array - input has {data.ndim}"

    # ensure that n empty array can't be an "optimal" split
    if data.size == 0:
        return np.inf

    # for classes of 1 and 0 get probabilities
    probability_ones = np.sum(data) / len(data)
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
        data = pd.read_csv(file)

    return data


def split_numeric(attribute: str,
                  value: float,
                  data: pd.DataFrame) -> tuple:
    """
    right_heavy split of numeric attribute  left -> < value, right >= value

    :param attribute: attribute to split on
    :param value: value to split by
    :param data: dataframe to split
    :return: tuple of resulting splits
    """

    assert np.issubdtype(data[attribute].dtype, np.number), 'selected column must be numeric'
    assert not any(df[attribute].isna()), 'split column cannot contain blank values, please fill before using split'

    split_below = data.loc[data[attribute] < value]
    split_above = data.loc[data[attribute] >= value]

    # discard attributes that are 'pure' - categorical attributes, essentially such as class
    if split_above[attribute].nunique() == 1:
        split_above = split_above.drop(columns=[attribute])

    if split_below[attribute].nunique() == 1:
        split_below = split_below.drop(columns=[attribute])

    return split_above, split_below


def split_non_numeric(attribute: str,
                      data: pd.DataFrame) -> tuple:
    """
    splits non-numeric data
    this results in MULTI-WAY splitting
    the splits do NOT contain the column that was split on (since after the split it is "pure" unlike numeric cols

    :param attribute: attribute that the data will be split on
    :param data: dataframe to split
    :return: tuple of split data
    """
    assert not np.issubdtype(data[attribute].dtype, np.number), 'selected attribute can\'t be numeric'
    assert not any(df[attribute].isna()), 'split column cannot contain blank values, please fill before using split'

    outcomes = data[attribute].unique()
    # filter for each possible outcome of the data
    parts = tuple(data.loc[data[attribute] == outcome] for outcome in outcomes)
    # for each part, discard the now-pure split column since it is useless
    parts = tuple(part.drop(columns=[attribute]) for part in parts)

    return parts


def information_gain(before_split: np.array,
                     splits: tuple,
                     target_column: str = t_col) -> float:
    """
    takes an original array and multiple sub-arrays and calculates information gain in bits based on entropy base 2

    :param before_split: array, 1 and 0
    :param splits: list of arrays, sub-arrays of before_splits
    :param target_column: str, dependent variable
    :return: float
    """

    assert len(before_split) == sum([len(part) for part in splits]), f"splits must add up to length of original{len(before_split)}/{sum([len(j) for j in splits])}"

    # get parameters all arrays - size, entropy
    # only for convenience / readability
    size_before = len(before_split)
    entropy_before = entropy(before_split[target_column])

    entropy_after = 0

    # add all the partial entropies of the sub-parts
    for part in splits:
        # get the size of the part and it's entropy
        part_size = len(part)
        entropy_split = entropy(part[target_column])

        # get the entropy contribution of the part, add it to entropy_before
        relative_size = part_size / size_before
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
    # apparently, with enough tree depth no attributes are left
    # shouldn't happen, solved with assert / split depth < attribute number

    if criterion == 'gini':
        pass

    split_attribute = None
    best_ig = 0
    split_value = 0
    best_splits = None
    leaf_sizes = 0
    numeric_split = None

    # first check if split on numeric / non-numeric
    for attribute in attributes:
        if np.issubdtype(data[attribute].dtype, np.number):

            # "inner loop" - split on each numeric value and test if information gain is better
            for outcome in data[attribute].unique():
                parts = split_numeric(attribute=attribute, value=outcome, data=data)
                ig = information_gain(before_split=data, splits=parts)

                # if a better IG is found, record it
                if ig > best_ig:
                    best_ig = ig
                    split_attribute = attribute
                    split_value = outcome
                    numeric_split = True
                    best_splits = parts
                    leaf_sizes = [part.shape[0] for part in parts]

        else:
            # non-numeric split - this supports multi-way-split
            parts = split_non_numeric(attribute=attribute, data=data)
            ig = information_gain(before_split=data, splits=parts)

            if ig > best_ig:
                best_ig = ig
                split_value = None
                split_attribute = attribute
                numeric_split = False
                best_splits = parts
                leaf_sizes = [part.shape[0] for part in parts]

    # hand over a dict for convenience
    result = {'value': split_value,
              'split_attribute': split_attribute,
              f'best_{criterion}': best_ig,
              'splits': best_splits,
              'numeric_split': numeric_split,
              'leaf_sizes': leaf_sizes}

    return result


def gini_index(before_split: pd.DataFrame,
               splits: tuple,
               outcomes: list,
               target_column: str = t_col) -> float:
    # get total elements to calculate shares
    total_elements = before_split.shape[0]

    gini = 0

    for split_data in splits:
        split_size = len(split_data) / total_elements

        if split_size == 0:
            continue

        score = 0

        for outcome in outcomes:
            # get fraction of cases where target == outcome
            p = np.mean(split_data[split_data[target_column] == outcome])
            score += p ** 2

        gini += (1.0 - score) * (split_size / total_elements)

    return gini


def create_leaf(part: pd.DataFrame,
                target_column: str = t_col) -> int:
    """
    create a terminal node / leaf
    :param part: data in the leaf
    :param target_column: column that is checked vor majority voting
    :return: int
    """

    # return the most frequent value (always 0/1) from the data to enter into the tree-dict
    most_frequent = part[target_column].mode()[0]

    return most_frequent


def split(node: dict,
          max_depth: int,
          min_size: int,
          depth: int = 1,
          target_column: str = t_col) -> None:
    """
    takes in a dict and modifies (recursively splits) it
    until max depth is reached or the sub-dicts (child nodes) are pure
    the dict is modified in place

    :param node: initial dict
    :param max_depth: maximum recursion depth (if nodes are not pure yet, majority vote (see create_leaf)
    :param min_size: minimum leaf-size
    :param depth: current depth, used to track recursion limit
    :param target_column:
    :return: None
    """

    # get the child splits out of the node-dict
    sub_nodes = node['splits']
    # records what the child-nodes were split on
    split_attribute = node['split_attribute']
    # bool - if the split was numeric or not - if False this may be a multiway-split
    node_split_numeric = node['numeric_split']
    # this is used for iterating over the child nodes
    number_nodes = len(sub_nodes)

    # free up memory from the original dict (otherwise data is duplicated while this runs
    del (node['splits'])

    if number_nodes == 1:
        # only one group is left, make a leaf
        node[0] = node[1] = create_leaf(pd.concat(sub_nodes))
        return

    # check if max_depth is reached
    if depth >= max_depth:
        # create a leaf for each group
        for i in range(number_nodes):
            # concatenate all the nodes into one, majority vote
            node[i] = create_leaf(sub_nodes[i])
        return

    # process all child nodes
    for i in range(number_nodes):
        # todo: logically create the dict key from i --> if numeric split: greater/smaller, else plain values
        child_node = sub_nodes[i]

        # check for minimum size
        if child_node.shape[0] <= min_size:
            node[i] = create_leaf(child_node)

        # check for pure node - if the node is pure, make it terminal
        elif child_node[target_column].nunique() == 1:
            node[i] = create_leaf(child_node)

        # if no exit case is reached, go one level lower
        else:
            node[i] = find_ideal_split(child_node)
            split(node[i], max_depth, min_size, depth=depth+1)


def grow_tree(data: pd.DataFrame,
              max_depth: int = 5,
              min_leaf_size: int = 10) -> dict:
    """
    take a dataframe with target, create the first split and then apply recursive split to it

    :param data: pd.DataFrame
    :param max_depth: maximum depth of the tree
    :param min_leaf_size: minimum samples per leaf
    :return: dictionary
    """

    assert max_depth < (data.shape[1] - 1), 'max depth of the tree must be less than attributes of the data '

    # start the tree off with the first split
    root = find_ideal_split(data)

    # recursion, see above
    split(root, max_depth, min_leaf_size, 1)

    return root


def predict_row(row: pd.DataFrame,
                tree: dict,
                data: pd.DataFrame) -> int:
    """
    create a prediction for a single row in the dataframe
    :param row: dataframe row
    :param tree: decision tree dict
    :param data: complete dataframe (to be replaced)
    :return: int 0/1
    """

    # check if value is None --> meaning it's a numeric attr
    # decide which "direction" to go
    traversal_attribute = tree['split_attribute']

    if not tree['value']:
        # for a numeric split, go in the direction that lines up with the value chosen
        # helper dict
        # todo: replace condition by numeric_split
        # todo: replace with direction by attribute value
        value_dic = {v: k for k, v in enumerate(data[traversal_attribute].unique())}
        direction = value_dic[row[traversal_attribute]]

    else:
        # for numeric values, go 0 for below, 1 for above (see split_numeric)
        direction = 0 if tree['value'] < row[traversal_attribute] else 1

    # check if we're at a leaf, if not recursion
    if isinstance(tree[direction], dict):
        return predict_row(row, tree[direction], data)
    else:
        return tree[direction]


def predict(data: pd.DataFrame,
            tree: dict) -> pd.Series:
    """
    convenience function, apply prediction to the df and return a series
    :param data: dataframe to predict on (format must be identical to training data)
    :param tree: decision tree dict
    :return: series of 0/1
    """

    predictions = data.apply(lambda x: predict_row(x, tree=tree, data=data), axis=1)

    return predictions


def evaluate_tree(data: pd.DataFrame,
                  tree: dict,
                  metric: str = 'accuracy',
                  target_column: str = t_col) -> dict:
    """
    evaluate tree on a metric
    :param data: data to evaluate on (must be same format as training data)
    :param tree: decision tree dict
    :param metric: str name of metric
    :param target_column: str name of target column
    :return: dictionary
    """

    assert metric.lower() in ('accuracy', 'specificity', 'recall'), 'metric name must be one of "specificity", "recall" or "accuracy"'

    ground_truth = np.array(data[target_column])
    prediction = np.array(predict(data, tree))

    if metric == 'accuracy':
        # basic accuracy formula
        value = np.mean(prediction == ground_truth)

    elif metric == 'recall':
        # how many of the actual positives did we catch
        true_positives = (prediction == ground_truth) * (prediction == 1)
        false_negatives = (prediction == 0) * (ground_truth == 1)
        value = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

    else:
        # how many of our positives are really positives
        true_negatives = (prediction == ground_truth) * (prediction == 0)
        false_negatives = (prediction == 0) * (ground_truth == 1)
        value = np.sum(true_negatives) / (np.sum(true_negatives) + np.sum(false_negatives))

    result = {'metric': metric,
              'value': value}

    return result


if __name__ == '__main__':

    df = load_data('./data/titanic.csv')

    # preprocess
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df.dropna(subset=['Embarked'], inplace=True)
    df.drop(columns=['Ticket', 'Name', 'Cabin', 'PassengerId'], inplace=True)

    # in order to iterate over rows by index (fill gaps)
    df.reset_index(inplace=True)

    # start off with the initial split and grow the tree
    best_initial = find_ideal_split(df)
    test_tree = grow_tree(df, 5, 10)
    pprint(test_tree)

    print('evaluation test: \n ---------------------------')

    for eval_metric in ('specificity', 'accuracy', 'recall'):
        res = evaluate_tree(df, test_tree, metric=eval_metric)
        print(f'tree {res["metric"]}: {res["value"]}')
