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
        self.max_depth = max_depth

        assert target_column in self.data.columns, "dependent column not found in data"
        assert len(self.outcomes) == 2, "classification target must have exactly two outcomes"

    def information_gain(self,
                         attribute: str,
                         data: pd.DataFrame = None) -> float:
        """
        find information gain for an attribute of the data
        :param attribute: attribute to split the data on
        :param data: used for recursion (takes a dataframe)
        :return: dictionary
        """

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
                   data: pd.DataFrame = None) -> dict:
        """
        check information gain for all columns that are left, get the largest, return with name and IG
        :return: dict
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

    def filter_data(self,
                    attribute: str,
                    value: str,
                    data: pd.DataFrame = None) -> pd.DataFrame:
        """
        filters data by an attribute and value
        :param attribute: attribute to filter on
        :param value: outcome to filter on
        :param data: used for recursion
        :return: pd.dataFrame
        """
        if data is None:
            data = self.data
        filtered = data.loc[self.data[attribute] == value].reset_index(drop=True)

        # make sure to drop the filtered-on column, since it's "pure" now
        filtered.drop(columns=[attribute], inplace=True)
        return filtered

    def find_rules(self,
                   data: pd.DataFrame = None,
                   depth: int = 0):
        """
        recursively create rules for classification
        :param data: in recursion, uses a subset of the data
        :param depth: integer to limit stack depth in recursion
        :return: dictionary of rules
        """

        # if no data is input, use init data
#        if data is None:
#            return

        # find attribute with most IG to split on
        split_options = self.next_split(data)

        # if there are no attributes left to split on --> fin
        if split_options['information_gain'] == 0:
            # error - this leads to an edge case in 3 different ways
            # example: test data index 13, no more attributes to split on but 50/50 impure data
            # todo: implement majority-voting in leaves
            # todo: change recursion from looking forward (i.e. on next_split)
            # todo: implement tie breakers as mentioned here:
            # https://iopscience.iop.org/1748-3190/12/1/011004/media/bbaa416dsd.pdf
#            print(data.loc[0, 'go_out'])
#            print(data)
#            print(data.columns)
            # simply grab the first column
            pseudo_column = data.columns.to_list()[0]
            print(pseudo_column)
            # associate with first value
            pseudo_value = data.loc[0, pseudo_column]
            pseudo_target = data.loc[0, self.target_column]
            print(data.head())
            pseudo_rule = {pseudo_column: {pseudo_value: pseudo_target}}
            print(pseudo_rule)
            return pseudo_rule

        elif split_options['attribute'] is None:
            # todo: implement majority voting here
            return

        split_attribute = split_options['attribute']

        # list of unique outcomes in the split attribute
        outcomes = data[split_attribute].unique()

        # create dict
        rules = dict()
        rules[split_attribute] = dict()

        # iterate over all outcomes in the split attribute, filter data and check for purity
        for outcome in outcomes:
            filtered = self.filter_data(split_attribute, outcome, data)
#            print(filtered.head())
            values = filtered[self.target_column].unique()

            # if only one outcome remains (target column is pure) create a rule
            if len(values) == 1:
                rules[split_attribute][outcome] = values[0]

            # if outcome is not pure yet - recursion
            elif self.max_depth is None or depth < self.max_depth:
                rules[split_attribute][outcome] = self.find_rules(filtered,
                                                                  depth=depth+1)

        return rules

    def fit(self):
        """
        calls the find_rules method and sets rules into the rules_ parameter
        :return: None
        """
        rules = self.find_rules(self.data)
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
                target_column='go_out',
                max_depth=None
                )

#    print(t.next_split())
#    print(t.data.head())
#    t.find_rules()
    t.fit()
    pprint(t.rules_)
#    print(t.next_split(t.filter_data('forecast', 'rain')))
#    d = t.filter_data('forecast', 'sunny')
#    print(d)
#    print(t.next_split(data=d))

#    print(test_data.loc[1])

#    print(type(test_data.loc[1]))
#    print(t.rules_.keys())

#    first_key = list(t.rules_.keys())[0]
#    first_value = t.rules_[first_key]

#    print(test_data.loc[1, first_key])
#    print(first_value)

#    second_key = test_data.loc[1, first_key]
#    print(second_key)

#    print(t.rules_[first_key][second_key])

#    rule_dict = t.rules_

    def get_prediction(rule_dict,
                       row: pd.Series):

        if isinstance(rule_dict, dict):
            column = list(rule_dict.keys())[0]

#            print(column)

#            print(rule_dict[column])
            rule_dict = rule_dict[column][row[column]]

            return get_prediction(rule_dict, row)

        else:
            return rule_dict


#    test_data['predictions'] = test_data.apply(lambda x: get_prediction(rule_dict=t.rules_, row=x), axis=0)

    print(test_data.loc[13])
    print(get_prediction(t.rules_, test_data.loc[13]))
