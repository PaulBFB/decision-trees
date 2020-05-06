import numpy as np
import pandas as pd


class ID3Tree:
    def __init__(self,
                 data: pd.DataFrame,
                 nodes: dict = None):
        # initializer, blank placeholder attributes
        self.data = data
        self.nodes = nodes

    def entropy(self,
                column: str):
        assert column in self.data.columns, "column not found in "

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
    t = ID3Tree(data=test_data)

    print(t)