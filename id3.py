import numpy as np
import pandas as pd
from information_gain import entropy #, information_gain
from math import log2


class ID3Tree:
    def __init__(self,
                 data: pd.DataFrame,
                 target_column: str = None,
                 nodes: dict = None):

        self.data = data
#        self.X = data.drop(columns=filter(lambda x: x != target_column, data.columns))
        self.nodes = nodes
        self.target_column = target_column
#        self.y = data[target_column].values
        self.outcomes = data[target_column].unique()

#        assert target_column in self.data.columns, "dependent column not found in data"
        assert len(self.outcomes) == 2, "classification target must have exactly two outcomes"

    def information_gain(self,
                         attribute):
        outcomes = self.data[attribute].unique()
        conditional_entropy = 0
        for possibility in outcomes:
            part = self.data[self.data[attribute] == possibility]
            # get entropy of target after split
            e = entropy(part[self.target_column] == self.outcomes[0])
            # get size of column in relation to unsplit data
            frac = part.shape[0] / self.data.shape[0]
            conditional_entropy += e * frac

        # compare conditional entropy of attribute to entropy of entire dataset
        ig = entropy(self.data[self.target_column] == self.outcomes[0]) - conditional_entropy

        return ig
#        conditional_entropy =

    def next_split(self):
        """
        check information gain for all columns that are left, get the largest, return with name and IG
        :return: dict()
        """
        split_attribute = {'attribute': None,
                           'information_gain': 0}
        for attribute in filter(lambda x: x != self.target_column, self.data.columns):
            potential_ig = self.information_gain(attribute)
            if potential_ig > split_attribute['information_gain']:
                split_attribute['attribute'] = attribute
                split_attribute['information_gain'] = self.information_gain(attribute)
#        information_gains = {key: self.information_gain(key) for key in self.data.columns if key != self.target_column}
        return split_attribute

    def split(self):
        """
        splits data on attribute of next_split (max information gain) - get all unique outcomes of attribute and return
        list of dataframes (one for each outcome)
        :return: list of pd.DataFrame
        """
        # get name of the attribute to split on
        split_on = self.next_split()['attribute']
        # initialize empty list to append onto
        parts = list()
        # get all possible outcomes, split with .loc method (in order to return copies
        for outcome in self.data[split_on].unique():
            parts.append(self.data.loc[self.data[split_on] == outcome])

        # todo: drop the column that the data was split on for recursion??
        # todo: change output into dict with {[attribute|outcome]: [DataFrame]}

        return parts

    def fit(self,
            dependent_column: str):
        pass
        # recursive fit function - check all attributes, get information gain, split on largest OIG
        # base cases: leaves are pure / no more attributes / no information gain (?)

    def score(self,
#              data: pd.DataFrame,
              dependent_column: str):
        pass
        # calculate accuracy of results

    def show(self):
        pass
        # representation of the tree as decision graph

    def __repr__(self):
        return f'id3 decision tree object - nodes: {"not fit yet" if not self.nodes else self.nodes}'
        # string representation of tree


if __name__ == '__main__':
    with open('./weather_decision.csv', mode='r') as file:
        test_data = pd.read_csv(file)
    t = ID3Tree(data=test_data,
                target_column='go_out')

    print(test_data.head())

    print(t)
    for i in ['forecast', 'temperature', 'humidity', 'windy']:
        print(t.information_gain(i))

    print(t.next_split())
    parts_after_first_split = t.split()
    for p in parts_after_first_split:
        print(p.head())
