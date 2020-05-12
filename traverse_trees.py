import pandas as pd
from pprint import pprint


with open('/home/paul/PycharmProjects/decision-trees/weather_decision.csv', mode='r') as file:
    test_data = pd.read_csv(file)


def purify_df(data: pd.DataFrame,
              target_col: str,
              depth: int = 0,
              max_depth: int = None,
              rules: list = [],
              paths_visited: dict = None):

    cols = list(filter(lambda x: x != target_col, data.columns))

    if len(cols) == 0:
        return rules

    outcomes = data[cols[0]].unique()

    filtered = data.loc[data[cols[0]] == outcomes[0]]
    rules.append(f'{cols[0]}=={outcomes[0]}')

    if all(filtered[target_col] == 'yes'):
        return rules
    else:
        return purify_df(data.drop(columns=[cols[0]]),
                         target_col=target_col,
                         rules=rules)

#    return True

for i in test_data.columns:
    print(f'column: {i} | outcomes: {test_data[i].unique()}')
test = purify_df(data=test_data.drop(columns=['forecast']), target_col='go_out')
#print(test_data.loc[test_data['forecast'] == 'overcast'])
#print(test)

r = {}
filtered_data = test_data
while not all(filtered_data['go_out'] == 'yes') and not all(filtered_data['go_out'] == 'no'):
    for i in filter(lambda x: x != 'go_out', test_data.columns):
        print(i)
        for j in test_data[i].unique():
            query = f'{i}=="{j}"'
            r[query] = filtered_data.query(query).shape
            filtered_data = test_data.query(query)

pprint(r)


def purify(data,
           target_column,
           visited,
           paths: list = None):

    if not paths:
        paths = []

    # columns left to visit are all unvisited
    to_visit_cols = list(filter(lambda x: x not in visited and x != target_column, data.columns))
    # if none are left to visit, exit

    if len(to_visit_cols) == 0:
        return paths, visited

    # take next column to visit, record that it has been visited
    next_col = to_visit_cols[0]
    visited.append(next_col)


    # get all outcomes values for the column, build all possible queries from it
    outcomes = data[to_visit_cols[0]].unique()
    queries = [f'{next_col}=="{i}"' for i in outcomes]

    # filter queries, by paths already taken
    queries = list(filter(lambda x: x not in visited, queries))

    if all(data[target_column] =='yes') or all(data[target_column] =='no'):
        paths += queries#[0]
    else:
        return purify(data.query(queries[0]).drop(columns=[next_col]), target_column, visited, paths)

    return paths, visited


t = purify(test_data, 'go_out', [], [])
print(t)
