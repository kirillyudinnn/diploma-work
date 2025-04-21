import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def get_matrix_interactions(df_train: pd.DataFrame, user_colname: str, item_colname: str):
    ALL_USERS = df_train[user_colname].unique().tolist()
    ALL_ITEMS = df_train[item_colname].unique().tolist()

    user_ids = dict(list(enumerate(ALL_USERS)))
    item_ids = dict(list(enumerate(ALL_ITEMS)))

    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    df_train['user_id'] = df_train[user_colname].map(user_map)
    df_train['item_id'] = df_train[item_colname].map(item_map)

    row = df_train['user_id'].values
    col = df_train['item_id'].values
    data = np.ones(df_train.shape[0])
    csr_train = csr_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))

    return csr_train


def prepare_data(
        X: pd.DataFrame, 
        user_colname: str, 
        item_colname: str, 
        min_interactions: int, 
        max_intercations: int,
        min_item_freq: int = None
) -> pd.DataFrame:
    user_intercations = X[user_colname].value_counts()
    users_to_keep = user_intercations[user_intercations.between(min_interactions, max_intercations)].index
    X = X[X[user_colname].isin(users_to_keep)]

    if min_item_freq:
        item_freq = X[item_colname].value_counts()
        items_to_keep = user_intercations[item_freq >= min_item_freq].index
        X = X[X[item_colname].isin(items_to_keep)]

    return X
    

def train_val_split(
        X: pd.DataFrame,
        val_k_items: int,
        test_k_items: int
):
    pass