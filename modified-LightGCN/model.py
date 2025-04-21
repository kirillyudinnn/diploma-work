import torch
import torch.nn as nn
import numpy as np

from typing import Dict


class LightGCNModel(nn.Module):
    def __init__(self, CONFIG: Dict):
        """
            Config keys:
                data: Interactions matrix
                n_users: Number of users
                n_items: Number of items
                r: r-AdjNorm factor, must be > 0
                n_layers: Number of message passing 
                emb_dim: Embedding size
                device: torch.device
            -param CONFIG: model configuration
              
        """

        super(LightGCNModel, self).__init__()
        self.n_users = CONFIG["n_users"]
        self.n_items = CONFIG["n_items"]
        self.emb_dim = CONFIG["emb_dim"]
        self.n_layers = CONFIG["n_layers"]
        self.r = CONFIG['r']
        self.device = torch.device(CONFIG["device"])

        self._init_embeddings()
        self.A_tilda = self._get_A_tilda(CONFIG["data"]).to(self.device)

    def _init_embeddings(self):
        self.E0 = nn.Embedding(self.n_users + self.n_items, self.emb_dim).to(self.device)
        nn.init.xavier_uniform_(self.E0.weight)
        self.E0.weight = nn.Parameter(self.E0.weight).to(self.device)

    def propagate_through_layers(self):
        """ Performs Light Graph Convolution """
        E_lyr = self.E0.weight

        for layer in range(self.n_layers):
            E_lyr = torch.add(
                E_lyr, 
                torch.sparse.mm(self.A_tilda, E_lyr)
            )

        E_lyr /= (self.n_layers + 1)

        final_user_Embed, final_item_Embed = torch.split(E_lyr, [self.n_users, self.n_items])

        return final_user_Embed, final_item_Embed

    def forward(self, users, pos_items, neg_items):
        usr_embeds, itm_embeds = self.propagate_through_layers()
        usr_embeds, pos_embeds = usr_embeds[users], itm_embeds[pos_items]
        neg_embeds = itm_embeds[neg_items]

        return usr_embeds, pos_embeds, neg_embeds

    def _get_A_tilda(self, data):
        indices = torch.LongTensor(np.array([data['user_idx'].values, data['item_idx'].values]))
        values = torch.FloatTensor([1.0] * len(data))

        R = torch.sparse.FloatTensor(indices, values, torch.Size([self.n_users, self.n_items]))

        l_u_indices = torch.LongTensor(np.array([data['user_idx'].values, data['user_idx'].values]))
        l_u_values = torch.FloatTensor([0.0] * len(data))
        left_up_zero = torch.sparse.FloatTensor(l_u_indices, l_u_values, torch.Size([self.n_users, self.n_users]))

        r_d_indices = torch.LongTensor(np.array([data['item_idx'].values, data['item_idx'].values]))
        r_d_values = torch.FloatTensor([0.0] * len(data))
        right_down_zero = torch.sparse.FloatTensor(r_d_indices, r_d_values, torch.Size([self.n_items, self.n_items]))

        upper_part = torch.cat((left_up_zero, R), 1)
        down_part = torch.cat((R.t(), right_down_zero), 1)
        adj_mat = torch.vstack((upper_part, down_part))

        rowsum = adj_mat.sum(1)

        d_inv_left = torch.pow(1e-9 + rowsum.to_dense(), -self.r)
        d_inv_left[torch.isinf(d_inv_left)] = 0.0
        offsets = torch.zeros((1,), dtype=torch.long)
        d_mat_inv_left = torch.sparse.spdiags(
                diagonals=d_inv_left,
                offsets=offsets,
                shape=(self.n_users + self.n_items, self.n_users + self.n_items)
        )

        d_inv_right = torch.pow(1e-9 + rowsum.to_dense(), -(1 - self.r))
        d_inv_right[torch.isinf(d_inv_left)] = 0.0
        offsets = torch.zeros((1,), dtype=torch.long)
        d_mat_inv_right = torch.sparse.spdiags(
                diagonals=d_inv_right,
                offsets=offsets,
                shape=(self.n_users + self.n_items, self.n_users + self.n_items)
            )

        norm_adj_mat = torch.sparse.mm(torch.sparse.mm(d_mat_inv_left, adj_mat), d_mat_inv_right)
        return norm_adj_mat
