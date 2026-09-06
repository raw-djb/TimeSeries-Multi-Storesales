import re
import pandas as pd
import numpy as np


def matching_pairs_within_column_name(df, column_name, scorer):
    from fuzzywuzzy import process
    unique_col_values = df[column_name].unique()
    score_sort = [(x,) + i
                  for x in unique_col_values
                  for i in process.extract(x, unique_col_values, scorer=scorer)]

    similarity_sort = pd.DataFrame(score_sort, columns=[column_name, 'match_sort', 'score_sort'])
    similarity_sort['sorted_' + column_name] = np.minimum(similarity_sort[column_name], similarity_sort['match_sort'])

    high_score_sort = similarity_sort[(similarity_sort['score_sort'] >= 80) &
                                       (similarity_sort[column_name] != similarity_sort['match_sort']) &
                                       (similarity_sort['sorted_' + column_name] != similarity_sort['match_sort'])]

    high_score_sort = high_score_sort.drop('sorted_' + column_name, axis=1).copy()

    return high_score_sort.groupby([column_name, 'score_sort']).agg(
        {'match_sort': ', '.join}).sort_values(['score_sort'], ascending=False)


def name_correction(x):
    x = x.lower()
    x = x.partition('[')[0]
    x = x.partition('(')[0]
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
    x = x.replace('  ', ' ')
    x = x.strip()
    return x


def lag_feature(df, lags, cols):
    for col in cols:
        tmp = df[["date_block_num", "shop_id", "item_id", col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_" + str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df


def categorize_rare(series, threshold=2, other_label="other"):
    counts = series.value_counts()
    keep = counts[counts >= threshold].index
    return series.apply(lambda x: x if x in keep else other_label)
