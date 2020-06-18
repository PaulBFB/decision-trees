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
                   depth: int = 0) -> dict:
        """
        recursively create rules for classification
        :param data: in recursion, uses a subset of the data
        :param depth: integer to limit stack depth in recursion
        :return: dictionary of rules
        """

        # find attribute with most IG to split on
        split_options = self.next_split(data)

        # if there are no attributes left to split on --> fin
        if split_options['information_gain'] == 0:
            # simply grab the first column
            pseudo_column = data.columns.to_list()[0]
            # associate with first value
            pseudo_value = data.loc[0, pseudo_column]
            pseudo_target = data.loc[0, self.target_column]
            pseudo_rule = {pseudo_column: {pseudo_value: pseudo_target}}

            p_rules = {i: data.loc[0, i] for i in filter(lambda x: x != self.target_column, data.columns)}

            n_rules = dict()
            for k, v in p_rules.items():
                n_rules[k] = {v: pseudo_target}
#            print(n_rules)
#            print(pseudo_rule)
            # error - this leads to an edge case in 3 different ways
            # example: test data index 13, no more attributes to split on but 50/50 impure data
            # note - test data in 13 still produces the error - other edge cases are solved (check with dummy text)
#            print(data)
#            print('exit case reached -- NO MORE IG')
#            print(n_rules)
            return n_rules # data.loc[0, self.target_column]

        elif split_options['attribute'] is None:
            print('exit case reached -- no more attributes left ')
            return data[self.target_column].mode()[0]

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
            number_outcomes = filtered[self.target_column].nunique()

            # if only one outcome remains (target column is pure) create a rule
            if number_outcomes == 1:
                rules[split_attribute][outcome] = filtered.loc[0, self.target_column]

            # if outcome is not pure yet - recursion
            elif self.max_depth is None or depth <= self.max_depth:
                rules[split_attribute][outcome] = self.find_rules(filtered,
                                                                  depth=depth+1)

#            else:
#                return filtered[self.target_column].mode()[0]

        return rules

    def split_new(self,
                  data: pd.DataFrame = None,
                  max_depth: int = 3,
                  depth: int = 0,
                  existing_rules: dict = None,
                  path: str = None):

        next_split = self.next_split(data)

        if existing_rules is None:
            existing_rules = dict()

        # exit condition 1 - pure value
        if data[self.target_column].nunique() == 1:
            return data.loc[0, self.target_column]

        # exit condition 2 - no more information can be gained
        elif next_split['information_gain'] == 0:
            # break the tie by picking the first value
            return data.loc[0, self.target_column]

        # exit condition 3 - no more attributes are left
        # does this case exist? no more IG, no attribute, but not pure?
        elif next_split['attribute'] is None:
            return 'WARNING CASE!!!'

        # exit condition 4 - max depth has been reached, but leaves are not pure
        elif depth == max_depth:
            # majority vote
            return data[self.target_column].mode()[0]

        # recursion - split further down
        else:
            split_attribute = next_split['attribute']
            possible_outcomes = data[split_attribute].unique()

            for outcome in possible_outcomes:
                print(depth, split_attribute, outcome)
                # get all possible outcomes of the attribute
                subset = self.filter_data(split_attribute, outcome, data)
            #    print(subset.shape)
                existing_rules[split_attribute] = dict()
#                existing_rules[split_attribute] = existing_rules.get(split_attribute, dict())
#                existing_rules[split_attribute][outcome] = self.split_new(data=subset,
#                                                                          depth=depth+1,
#                                                                          existing_rules=existing_rules)
                existing_rules[split_attribute][outcome] = existing_rules[split_attribute].get(outcome, self.split_new(data=subset,
                                                                                                                       depth=depth+1,
                                                                                                                       existing_rules=existing_rules))
#                existing_rules[split_attribute][outcome] = existing_rules
            return existing_rules

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
    with open('../data/weather_decision.csv', mode='r') as file:
        test_data = pd.read_csv(file)
    t = ID3Tree(data=test_data,
                target_column='go_out',
                max_depth=None)


    def get_prediction(rule_dict,
                       row: pd.Series):

        if not isinstance(rule_dict, dict):
            return rule_dict

        else:
            column = list(rule_dict.keys())[0]
            rule_dict = rule_dict[column][row[column]]
            return get_prediction(rule_dict, row)


#    rules_test = t.split_new(test_data, max_depth=5)
#    pprint(rules_test)

    t.fit()
    pprint(t.rules_)

    print(test_data.loc[8])
    prediction = get_prediction(t.rules_, test_data.loc[8])
    print(prediction)

#    for i in range(test_data.shape[0]):
#        print(i)
#        print(test_data.loc[i])
#        prediction = get_prediction(t.rules_, test_data.loc[i])
#        print(prediction)
