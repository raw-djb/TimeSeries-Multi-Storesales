import os
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import name_correction, lag_feature, categorize_rare

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


def load_raw():
    sales_train_df = pd.read_csv(os.path.join(RAW_DIR, "sales_train.csv"))
    items_categories_df = pd.read_csv(os.path.join(RAW_DIR, "item_categories.csv"))
    items_df = pd.read_csv(os.path.join(RAW_DIR, "items.csv"))
    shops_df = pd.read_csv(os.path.join(RAW_DIR, "shops.csv"))
    test_df = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))
    return sales_train_df, items_categories_df, items_df, shops_df, test_df


def clean_sales(sales_train_df):
    sales_train_df = sales_train_df[sales_train_df.item_cnt_day < 900]
    sales_train_df = sales_train_df[sales_train_df.item_price < 300000]
    sales_train_df["revenue"] = sales_train_df["item_cnt_day"] * sales_train_df["item_price"]
    return sales_train_df


def process_shops(shops_df):
    shops_df["shop_name_split_1"] = shops_df.shop_name.str.split(" ").map(lambda x: x[0])
    shops_df["shop_name_split_2"] = shops_df.shop_name.str.split(" ").map(lambda x: x[1])

    shops_df.loc[shops_df.shop_name_split_1 == "!Якутск", "shop_name_split_1"] = "Якутск"
    shops_df["shop_name_split_1"] = categorize_rare(shops_df.shop_name_split_1)
    shops_df["shop_name_split_2"] = categorize_rare(shops_df.shop_name_split_2)

    shops_df["shop_name_split_1_encode"] = LabelEncoder().fit_transform(shops_df.shop_name_split_1)
    shops_df["shop_name_split_2_encode"] = LabelEncoder().fit_transform(shops_df.shop_name_split_2)
    shops_df = shops_df[["shop_id", "shop_name_split_1_encode", "shop_name_split_2_encode"]]
    return shops_df


def process_categories(items_categories_df):
    fix_map = {
        'PC - Гарнитуры/Наушники': "Гарнитуры/Наушники - PC",
        'Игры PC - Дополнительные издания': "Дополнительные издания - Игры PC",
        'Игры PC - Коллекционные издания': "Коллекционные издания - Игры PC",
        'Игры PC - Стандартные издания': "Стандартные издания - Игры PC",
        'Игры PC - Цифра': "Цифра - Игры PC",
    }
    for old, new in fix_map.items():
        items_categories_df.loc[items_categories_df.item_category_name == old, "item_category_name"] = new

    items_categories_df["item_cat_split_1"] = items_categories_df.item_category_name.str.split("-").str[0]
    items_categories_df["item_cat_split_2"] = items_categories_df.item_category_name.str.split("-").str[1].fillna("unknown")

    items_categories_df["item_cat_split_1"] = categorize_rare(items_categories_df.item_cat_split_1)
    items_categories_df["item_cat_split_2"] = categorize_rare(items_categories_df.item_cat_split_2)

    items_categories_df["item_cat_split_1_encode"] = LabelEncoder().fit_transform(items_categories_df.item_cat_split_1)
    items_categories_df["item_cat_split_2_encode"] = LabelEncoder().fit_transform(items_categories_df.item_cat_split_2)
    items_categories_df = items_categories_df[["item_category_id", "item_cat_split_1_encode", "item_cat_split_2_encode"]]
    return items_categories_df


def process_items(items_df):
    items_df[["name1", "name2"]] = items_df["item_name"].str.split(
        "[", n=1, regex=False, expand=True
    )

    items_df[["name1", "name3"]] = items_df["item_name"].str.split(
        "(", n=1, regex=False, expand=True
    )

    items_df["name2"] = items_df.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ", regex=True).str.lower()
    items_df["name3"] = items_df.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ", regex=True).str.lower()

    items_df = items_df.fillna('0')

    items_df["item_name"] = items_df["item_name"].apply(name_correction)
    items_df.name2 = items_df.name2.apply(lambda x: x[:-1] if x != "0" else "0")
    items_df["name2"] = items_df.name2.str.strip()

    items_df['type'] = items_df.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0])
    items_df.loc[items_df['type'] == '', 'type'] = 'pc'
    items_df.loc[(items_df.type == "x360") | (items_df.type == "xbox360") | (items_df.type == "xbox 360"), "type"] = "xbox 360"
    items_df.loc[(items_df.type == 'pc') | (items_df.type == 'рс') | (items_df.type == 'pс'), "type"] = "pc"

    group_item_sum = items_df.groupby(["type"]).agg({"item_id": "count"}).reset_index()
    drop_cols = []
    for cat in group_item_sum.type.unique():
        if group_item_sum.loc[(group_item_sum.type == cat), "item_id"].values[0] <= 20:
            drop_cols.append(cat)
    items_df.type = items_df.type.apply(lambda x: "other" if (x in drop_cols) else x)

    items_df.type = LabelEncoder().fit_transform(items_df.type)
    items_df.name3 = LabelEncoder().fit_transform(items_df.name3)
    items_df.drop(["item_name", "name1", "name2"], axis=1, inplace=True)
    return items_df


def build_matrix(sales_train_df, test_df):
    cols = ["date_block_num", "shop_id", "item_id"]
    matrix = []
    for i in range(34):
        sales = sales_train_df[sales_train_df.date_block_num == i]
        matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype=np.int16))

    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
    matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
    matrix["item_id"] = matrix["item_id"].astype(np.int16)
    matrix.sort_values(cols, inplace=True)

    group = sales_train_df.groupby(cols).agg({"item_cnt_day": ["sum"]})
    group.columns = ["item_cnt_month"]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=cols, how="left")
    matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0).astype(np.float16)

    test_df = test_df.copy()
    test_df["date_block_num"] = 34
    test_df["date_block_num"] = test_df["date_block_num"].astype(np.int8)
    test_df["shop_id"] = test_df.shop_id.astype(np.int8)
    test_df["item_id"] = test_df.item_id.astype(np.int16)

    matrix = pd.concat([matrix, test_df.drop(["ID"], axis=1)], ignore_index=True, sort=False, keys=cols)
    matrix.fillna(0, inplace=True)
    return matrix


def merge_descriptive(matrix, shops_df, items_df, items_categories_df):
    matrix = pd.merge(matrix, shops_df, on=["shop_id"], how="left")
    matrix = pd.merge(matrix, items_df, on=["item_id"], how="left")
    matrix = pd.merge(matrix, items_categories_df, on=["item_category_id"], how="left")

    matrix["shop_name_split_1_encode"] = matrix["shop_name_split_1_encode"].astype(np.int8)
    matrix["shop_name_split_2_encode"] = matrix["shop_name_split_2_encode"].astype(np.int8)
    matrix["item_cat_split_1_encode"] = matrix["item_cat_split_1_encode"].astype(np.int8)
    matrix["item_cat_split_2_encode"] = matrix["item_cat_split_2_encode"].astype(np.int8)
    matrix["item_type"] = matrix["type"].astype(np.int8)
    matrix["item_name3"] = matrix["name3"].astype(np.int16)
    matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)

    matrix.drop(["type", "name3"], axis=1, inplace=True)
    return matrix


def add_group_lag(matrix, group_cols, agg_col, new_col_name, lags):
    group = matrix.groupby(group_cols).agg({agg_col: ["mean"]})
    group.columns = [new_col_name]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=group_cols, how="left")
    matrix[new_col_name] = matrix[new_col_name].astype(np.float16)
    matrix = lag_feature(matrix, lags, [new_col_name])
    matrix.drop([new_col_name], axis=1, inplace=True)
    return matrix


def engineer_features(matrix, sales_train_df):
    matrix = lag_feature(matrix, [1, 2, 3], ["item_cnt_month"])

    matrix = add_group_lag(matrix, ['date_block_num'], 'item_cnt_month', 'date_avg_item_cnt', [1])
    matrix = add_group_lag(matrix, ['date_block_num', 'item_id'], 'item_cnt_month', 'date_item_avg_item_cnt', [1, 2, 3])
    matrix = add_group_lag(matrix, ['date_block_num', 'shop_id'], 'item_cnt_month', 'date_shop_avg_item_cnt', [1, 2, 3])
    matrix = add_group_lag(matrix, ['date_block_num', 'shop_id', 'item_id'], 'item_cnt_month', 'date_shop_item_avg_item_cnt', [1, 2, 3])
    matrix = add_group_lag(matrix, ['date_block_num', 'shop_id', 'item_cat_split_1_encode'], 'item_cnt_month', 'date_shop_itemcat1_avg_item_cnt', [1, 2, 3])
    matrix = add_group_lag(matrix, ['date_block_num', 'shop_id', 'item_cat_split_2_encode'], 'item_cnt_month', 'date_shop_itemcat2_avg_item_cnt', [1, 2, 3])
    matrix = add_group_lag(matrix, ['date_block_num', 'shop_id', 'item_category_id'], 'item_cnt_month', 'date_shop_itemcat_avg_item_cnt', [1, 2, 3])
    matrix = add_group_lag(matrix, ['date_block_num', 'shop_id', 'item_type'], 'item_cnt_month', 'date_shop_itemtype_avg_item_cnt', [1, 2, 3])
    matrix = add_group_lag(matrix, ['date_block_num', 'shop_id', 'item_name3'], 'item_cnt_month', 'date_shop_itemname3_avg_item_cnt', [1, 2, 3])

    group = sales_train_df.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_avg_item_price']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['item_id'], how='left')
    matrix["item_avg_item_price"] = matrix.item_avg_item_price.astype(np.float16)

    group = sales_train_df.groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})
    group.columns = ['date_avg_item_price']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
    matrix.date_avg_item_price = matrix["date_avg_item_price"].astype(np.float16)
    matrix = lag_feature(matrix, [1, 2, 3], ['date_avg_item_price'])
    matrix.drop(['date_avg_item_price'], axis=1, inplace=True)

    matrix = matrix[matrix.date_block_num > 2]
    return matrix


def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    sales_train_df, items_categories_df, items_df, shops_df, test_df = load_raw()

    sales_train_df = clean_sales(sales_train_df)
    shops_df = process_shops(shops_df)
    items_categories_df = process_categories(items_categories_df)
    items_df = process_items(items_df)

    matrix = build_matrix(sales_train_df, test_df)
    matrix = merge_descriptive(matrix, shops_df, items_df, items_categories_df)
    matrix = engineer_features(matrix, sales_train_df)

    matrix.to_pickle(os.path.join(PROCESSED_DIR, "data.pkl"))
    test_df.to_pickle(os.path.join(PROCESSED_DIR, "test_index.pkl"))

    return matrix


if __name__ == "__main__":
    start = time.time()
    df = run_pipeline()
    print("matrix shape:", df.shape)
    print("elapsed seconds:", time.time() - start)
