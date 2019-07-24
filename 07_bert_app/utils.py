import random
import math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
import gluonnlp as nlp

class DotProductAttention(nn.Block): 
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # mask: (batch_size, #queries, #kv_pairs)
    def forward(self, query, key, value, mask=None):
        d = query.shape[-1]
        # set transpose_b=True to swap the last two dimensions of key
        scores = nd.batch_dot(query, key, transpose_b=True) / math.sqrt(d)
        attention_weights = nlp.model.attention_cell._masked_softmax(mx.nd, scores, mask, scores.dtype)
        attention_weights = self.dropout(attention_weights)
        return nd.batch_dot(attention_weights, value)


class PositionalEncoding(gluon.nn.Block):
    def __init__(self, units, dropout=0, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self._max_len = max_len
        self._units = units
        self.embed = nn.Embedding(max_len, units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        pos_seq = mx.nd.arange(X.shape[1]).expand_dims(0)
        emb = self.embed(pos_seq)
        return self.dropout(X + emb)

class MultiHeadAttention(nn.Block):
    def __init__(self, units, num_heads, dropout, **kwargs):  # units = d_o
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert units % num_heads == 0
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Dense(units, use_bias=False, flatten=False)
        self.W_k = nn.Dense(units, use_bias=False, flatten=False)
        self.W_v = nn.Dense(units, use_bias=False, flatten=False)

    # query, key, and value shape: (batch_size, num_items, dim)
    # mask shape is (batch_size, query_length, memory_length)
    def forward(self, query, key, value, mask):
        # Project and transpose from (batch_size, num_items, units) to
        # (batch_size * num_heads, num_items, p), where units = p * num_heads.
        query, key, value = [transpose_qkv(X, self.num_heads) for X in (
            self.W_q(query), self.W_k(key), self.W_v(value))]
        if mask is not None:
            # Replicate mask for each of the num_heads heads
            mask = nd.broadcast_axis(nd.expand_dims(mask, axis=1),
                                    axis=1, size=self.num_heads)\
                    .reshape(shape=(-1, 0, 0), reverse=True)
        output = self.attention(query, key, value, mask)
        # Transpose from (batch_size * num_heads, num_items, p) back to
        # (batch_size, num_items, units)
        return transpose_output(output, self.num_heads)

def transpose_qkv(X, num_heads):
    # Shape after reshape: (batch_size, num_items, num_heads, p)
    # 0 means copying the shape element, -1 means inferring its value
    X = X.reshape((0, 0, num_heads, -1))
    # Swap the num_items and the num_heads dimensions
    X = X.transpose((0, 2, 1, 3))
    # Merge the first two dimensions. Use reverse=True to infer
    # shape from right to left
    return X.reshape((-1, 0, 0), reverse=True)

def transpose_output(X, num_heads):
    # A reversed version of transpose_qkv
    X = X.reshape((-1, num_heads, 0, 0), reverse=True)
    X = X.transpose((0, 2, 1, 3))
    return X.reshape((0, 0, -1))

class PositionWiseFFN(nn.Block):
    def __init__(self, units, hidden_size, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.ffn_1 = nn.Dense(hidden_size, flatten=False)
        self.activation = nn.GELU()
        self.ffn_2 = nn.Dense(units, flatten=False)

    def forward(self, X):
        return self.ffn_2(self.activation(self.ffn_1(X)))

class AddNorm(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm()

    def forward(self, X, Y):
        return self.norm(self.dropout(Y) + X)

def position_encoding_init(max_length, dim):
    X = nd.arange(0, max_length).reshape((-1,1)) / nd.power(
            10000, nd.arange(0, dim, 2)/dim)
    position_weight = nd.zeros((max_length, dim))

    position_weight[:, 0::2] = nd.sin(X)
    position_weight[:, 1::2] = nd.cos(X)
    return position_weight


class EncoderBlock(gluon.nn.Block):
    def __init__(self, units, hidden_size, num_heads, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(units, num_heads, dropout)
        self.add_1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(units, hidden_size)
        self.add_2 = AddNorm(dropout)

    def forward(self, X, mask):
        Y = self.add_1(X, self.attention(X, X, X, mask))
        return self.add_2(Y, self.ffn(Y))
