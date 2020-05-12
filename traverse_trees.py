import pandas as pd
from pprint import pprint


with open('/home/paul/PycharmProjects/decision-trees/weather_decision.csv', mode='r') as file:
    test_data = pd.read_csv(file)


def purify(data,
           target_column,
           path: list = None):

    if path is None:
        path = list()

    data_pure = all(data[target_column] == 'yes') or all(data[target_column] == 'no')
    print(data_pure)
#    all_paths_taken: [f'' for i in filter(lambda x: x != target_column, data.columns)]
    # if data is already pure, do nothing
    if data_pure:
        print('pure')
#        print(paths)
        return path
#        return purify(data, target_column, paths)

    # if no columns except for the target are there/left to visit, do nothing
    elif len(list(filter(lambda x: x != target_column, data.columns))) == 0:
        print('no more cols left to search')
        return path

    else:
        # grab the first column to visit
        to_visit_cols = list(filter(lambda x: x != target_column, data.columns))
        next_col = to_visit_cols[0]
        print(f'splitting next on: {next_col}')

        # for the column, get all outcomes, build queries from it
        outcomes = data[to_visit_cols[0]].unique()
        queries = [f'{next_col}=="{i}"' for i in outcomes]
        # for the query, make sure that it has not been used before
        queries = list(filter(lambda x: x not in path, queries))
#        print(f'query: {queries[0]}')

        # if no outcomes for the column that were picked are left, drop it and recurse
        if len(queries) == 0:
            return purify(data.drop(columns=[next_col]),
                          target_column=target_column,
                          path=path)
        else:
            path.append(queries[0])

        print(path)

        return purify(data.query(queries[0]), target_column, path)


print(test_data.head(20))
t = purify(test_data, 'go_out')
#print(t)

