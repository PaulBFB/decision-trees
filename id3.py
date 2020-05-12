import numpy as np
import pandas as pd
from information_gain import entropy #, information_gain
from math import log2
from pprint import pprint


class ID3Tree:
    def __init__(self,
                 data: pd.DataFrame,
                 target_column: str = None,
                 max_depth: int = None):

        self.data = data
        self.nodes = {0: {'no_split': data}}
        self.rules_ = None
        self.target_column = target_column
        self.outcomes = data[target_column].unique()

        assert target_column in self.data.columns, "dependent column not found in data"
        assert len(self.outcomes) == 2, "classification target must have exactly two outcomes"

    def information_gain(self,
                         attribute: str,
                         data: pd.DataFrame = None
                         ):

        if data is None:
            data = self.data

        outcomes = data[attribute].unique()
        conditional_entropy = 0

        for value in outcomes:
            part = data[data[attribute] == value]
            # get entropy of target after split
            e = entropy(part[self.target_column] == self.outcomes[0])
            # get size of column in relation to unsplit data
            frac = part.shape[0] / data.shape[0]
            conditional_entropy += e * frac

        # compare conditional entropy of attribute to entropy of entire dataset
        ig = entropy(self.data[self.target_column] == self.outcomes[0]) - conditional_entropy

        return ig

    def next_split(self,
                   data: pd.DataFrame = None):
        """
        check information gain for all columns that are left, get the largest, return with name and IG
        :return: dict()
        """

        if data is None:
            data = self.data

        split_attribute = {'attribute': None,
                           'information_gain': 0}
        for attribute in filter(lambda x: x != self.target_column, data.columns):
            potential_ig = self.information_gain(attribute, data)
            if potential_ig > split_attribute['information_gain']:
                split_attribute['attribute'] = attribute
                split_attribute['information_gain'] = self.information_gain(attribute, data)

        return split_attribute

    def split(self):
        """
        splits data on attribute of next_split (max information gain) - get all unique outcomes of attribute and return
        list of dataframes (one for each outcome)
        :return: dict with forma {attribute|ouctome: dataframe}
        """
        # get name of the attribute to split on
        split_on = self.next_split()['attribute']
        # initialize empty list to append onto
        parts = dict()
        # get all possible outcomes, split with .loc method (in order to return copies
        for outcome in self.data[split_on].unique():
            parts[f'{split_on}|{outcome}'] = self.data.loc[self.data[split_on] == outcome].drop(columns=[split_on])

        self.data.drop(columns=[split_on], inplace=True)
        # does that make sense - what if subsequent splits are identical? duplicate dict keys?

        return parts

    def filter_data(self,
                    attribute,
                    value,
                    data=None):
        if data is None:
            data = self.data
        filtered = data.loc[self.data[attribute] == value].reset_index(drop=True)
        return filtered

    def find_rules(self,
                   data=None,
                   rules=None):

        if data is None:
            data = self.data

        split_attribute = self.next_split(data)['attribute']
#        print(split_attribute)

        outcomes = data[split_attribute].unique()
#        print(outcomes)

        if rules is None:
            rules = {}
            rules[split_attribute] = {}

        for outcome in outcomes:
            filtered = self.filter_data(split_attribute, outcome, data)
            values = filtered[self.target_column].unique()
            #            print(values)
            if len(values) == 1:
#                print(rules)
                rules[split_attribute][outcome] = values[0]
#                print(rules)
            else:
                rules[split_attribute][outcome] = self.find_rules(filtered)
#                                                                  rules)

        return rules
#        self.rules_ = rules

    def fit(self):
        rules = self.find_rules()
        self.rules_ = rules

    def score(self,
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

#    print(t.next_split())
#    print(t.data.head())
#    t.find_rules()
    t.fit()
    pprint(t.rules_)
#    print(t.filter_data('forecast', 'sunny'))
#    print(t.next_split(t.filter_data('forecast', 'rain')))
#    d = t.filter_data('forecast', 'sunny')
#    print(d)
#    print(t.next_split(data=d))
