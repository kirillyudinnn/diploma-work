import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def get_matrix_interactions(df: pd.DataFrame, user_colname: str, item_colname: str):
    ALL_USERS = df[user_colname].unique().tolist()
    ALL_ITEMS = df[item_colname].unique().tolist()

    user_ids = dict(list(enumerate(ALL_USERS)))
    item_ids = dict(list(enumerate(ALL_ITEMS)))

    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    df['user_id'] = df[user_colname].map(user_map)
    df['item_id'] = df[item_colname].map(item_map)

    row = df['user_id'].values
    col = df['item_id'].values
    data = np.ones(df.shape[0])
    csr_train = csr_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))

    return csr_train
