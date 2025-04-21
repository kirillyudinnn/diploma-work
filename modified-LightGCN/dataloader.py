import pandas as pd
import numpy as np


class Dataloader:
    """
        Dataloader class storages DataFrame with user-items interactions 
        and returns [usr], [pos_itm], [neg_itm] indexes 

            User-Item interactions DataFrame
        # -----------------------------------------
        # | user_idx (int) | item_idx (List[Int]) |
        # -----------------------------------------
        # |          0        | [0, 1, 30]        |
        # |          1        | [40, 50]          |
        # |          2        | [60, 70, 80, 90]  |
        # -----------------------------------------

        interected_items_df : DataFrame with user-item interactions
        sampler_mode : Negative sampling mode [uniform or item-pop popularity]
        n_items : number of items in dataset
        n_users : number of users in dataset
        items_popularity_dist : 'item_idx' value_counts()
        num_neg_p_itm : Number of negative items per one positive item
        popularity_smooth_coef : popularity distribution smoothing coefficient, (1/2) by default

        to_explode : set True for validation data loader 
                It returns not random, but all positive user-item interactions
    """
    def __init__(
            self, 
            interected_items_df: pd.DataFrame,
            n_items: int,
            n_users: int,
            items_popularity_dist: np.array,
            num_neg_p_itm: int,
            popularity_smooth_coef:float = 0.5,
            to_explode:bool=False
    ):
        self.data = interected_items_df
        self.n_items = n_items
        self.n_users = n_users
        self.items_popularity_dist = items_popularity_dist ** popularity_smooth_coef
        self.num_neg_p_itm = num_neg_p_itm
        self.to_explode = to_explode

    def __getitem__(
            self, 
            batch_users: np.array
    ):
        """
        Data loader returns arrays of (users, positive items, negative items) indexes
        Example:
        
        Users: [0, 1, 5, 19]
        Positive items: [50, 20, 10, 30]
        Negative items; [N random values] repeats num_neg_p_itm * len(pos_items) times

        batch_users : Indexes of users in the batch
    
        It returns not random, but all positive user-item interactions when to_explode is True

        Example: 
        Users: [0, 0, 1, 1, 1]
        Positive items: [50, 51, 20, 21, 50]
        Negative items: [N random values] repeats num_neg_p_itm * len(pos_items) times
        """
        
        return self.__get(batch_users)
    

    def __get(
            self,
            batch_users: np.array,
    ):
        df = self.data[
            self.data["user_idx"].isin(batch_users)
        ]
        items = np.arange(self.n_items)

        if self.to_explode:
            df = df.explode("item_idx")
            positive_items = df["item_idx"].values.astype(np.int64)
        else:
            positive_items = df["item_idx"].apply(lambda x: np.random.choice(x)).values


        users = df["user_idx"].values
        negative_items = []

        items_popularity = self.items_popularity_dist.copy()
        items_popularity[positive_items] = 0

        norm = np.sum(items_popularity)
        probs = items_popularity / norm

        negative_items = np.random.choice(items, self.num_neg_p_itm, p=probs)

        negative_items = np.tile(negative_items, len(positive_items))

        return users, positive_items, negative_items
