import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_out, proj_d, n_heads=3):
        super(Attention, self).__init__()
        self.norm_k = nn.LayerNorm(dim_k)
        self.norm_q = nn.LayerNorm(dim_q)
        self.dim_out = dim_out
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.dim_v = dim_k
        self.dim_q = dim_q
        self.proj_d = int(proj_d/n_heads)
        self.keys = nn.Linear(dim_k, proj_d, bias=True)
        self.queries = nn.Linear(dim_q, proj_d, bias=True)
        self.fc_out = nn.Linear(dim_k * n_heads, dim_out)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(dim_out)

    def forward(self, values, keys, queries):
        batch, n_value = values.shape[0], values.shape[1]
        n_query = queries.shape[1]
        # keys, values = self.norm_k(keys), self.norm_k(values)
        # queries = self.norm_q(queries)
        # project into some other space, so that they can be compared with.
        keys_projected = self.keys(keys)  # (N, key_len, dim_k) -> (N, key_len, proj_d)
        queries_projected = self.queries(queries)  # (N, query_len, dim_q) -> (N, query_len, proj_d)
        values_projected = values.unsqueeze(2)

        keys_projected = keys_projected.view((batch, n_value, self.n_heads, self.proj_d))
        # values_projected = values_projected.view((batch, n_value, self.n_heads, self.proj_d ))
        values_projected = values_projected.repeat(1, 1, self.n_heads, 1)
        queries_projected = queries_projected.view((batch, n_query, self.n_heads, self.proj_d))

        # Einsum does matrix mult.
        # energy: (N, query_len, n_heads, dim_k)
        energy = torch.einsum("nqhd,nkhd->nqhk", [queries_projected, keys_projected])

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for better stability
        # attention shape: (N, query_len, n_heads, dim_k)
        attention = torch.softmax(energy / (self.proj_d ** (1 / 2)), dim=2)

        # out after matrix multiply: (N, query_len, dim_k)
        out = torch.einsum("nqhk,nkhd->nqhd", [attention, values_projected])
        out = out.reshape((batch, n_query, -1))
        # take the first row (attended data for the target bus)

        # we then reshape the last dimension.
        # final out shape: (N, query_len, dim_out)
        out = self.relu(self.fc_out(out))
        return out

