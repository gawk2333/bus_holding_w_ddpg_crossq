import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, q_dim, kv_dim, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.Wq = nn.Parameter(torch.empty(size=(q_dim, out_features)))
        nn.init.xavier_uniform_(self.Wq.data, gain=1.)

        self.Wkv = nn.Parameter(torch.empty(size=(kv_dim, out_features)))
        nn.init.xavier_uniform_(self.Wkv.data, gain=1.)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, q, kv, adj):
        q_proj = torch.mm(q, self.Wq)
        kv_proj = torch.mm(kv, self.Wkv)
        att = torch.mm(q_proj, kv_proj.T)
        att = F.softmax(att, dim=1)
        h_prime = torch.matmul(att, kv_proj)

        # a_input = self._prepare_attentional_mechanism_input(Wh)
        # e = F.elu(torch.matmul(a_input, self.a).squeeze(2))
        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # attention = F.softmax(attention, dim=1)
        # h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'