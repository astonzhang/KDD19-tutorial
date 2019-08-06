import random
import math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
import gluonnlp as nlp

class DotProductAttention(nn.Block): 
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d = query.shape[-1]
        scores = nd.batch_dot(query, key, transpose_b=True) / math.sqrt(d)
        attention_weights = nlp.model.attention_cell._masked_softmax(nd, scores, mask, scores.dtype)
        attention_weights = self.dropout(attention_weights)
        return nd.batch_dot(attention_weights, value)

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

class MultiHeadAttention(nn.Block):
    def __init__(self, units, num_heads, dropout, **kwargs):  # units = d_o
        super().__init__(**kwargs)
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


def position_encoding_init(max_length, dim):
    X = nd.arange(0, max_length).reshape((-1,1)) / nd.power(
            10000, nd.arange(0, dim, 2)/dim)
    position_weight = nd.zeros((max_length, dim))

    position_weight[:, 0::2] = nd.sin(X)
    position_weight[:, 1::2] = nd.cos(X)
    return position_weight


class PositionalEncoding(nn.Block):
    def __init__(self, units, dropout=0, max_len=1000):
        super().__init__()
        self._max_len = max_len
        self._units = units
        self.position_weight = position_encoding_init(max_len, units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        pos_seq = nd.arange(X.shape[1]).expand_dims(0)
        emb = nd.Embedding(pos_seq, self.position_weight, self._max_len, self._units)
        return self.dropout(X + emb)

def print_side_by_side(*strings, sep='\t\t'):
    split = [str(s).split("\n") for s in strings]
    zipped = zip(*split)
    for elems in zipped:
        print(sep.join(elems))

