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

def train_loop(net, train_data, test_data, num_epoch, lr, ctx, loss_fn):
    trainer = gluon.Trainer(net.collect_params(), 'bertadam',
                            {'learning_rate': lr, 'wd':0.01},
                            update_on_kvstore=False)
    params = [p for p in net.collect_params().values() if p.grad_req != 'null']
    grad_clip = 1
    
    num_warmup_steps = 50
    step_num = 0
    num_train_steps = len(train_data) * num_epoch

    for epoch in range(num_epoch):
        accuracy = mx.metric.Accuracy()
        running_loss = 0
        for i, (inputs, seq_lens, token_types, labels) in enumerate(train_data):
            step_num += 1
    
            # learning rate schedule
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                non_warmup_steps = step_num - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset * lr
            trainer.set_learning_rate(new_lr)
            inputs = gluon.utils.split_and_load(inputs, ctx)
            seq_lens = gluon.utils.split_and_load(seq_lens, ctx)
            token_types = gluon.utils.split_and_load(token_types, ctx)
            labels = gluon.utils.split_and_load(labels, ctx)

            losses = []
            preds = [] 
            with mx.autograd.record():
                for inp, seq_len, token_type, label in zip(inputs, seq_lens, token_types, labels):
                    out = net(inp, token_type, seq_len)
                    loss = loss_fn(out, label.astype('float32'))
                    losses.append(loss)
                    preds.append(out)
            mx.autograd.backward(losses)
            for l in losses:
                running_loss += l.mean().asscalar() / len(losses)
            # Gradient clipping
            trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.update(1)
    
            accuracy.update(labels, preds)
            if i % 25 == 0:
                print("Batch", i, "Acc", accuracy.get()[1],"Train Loss", running_loss/(i+1))
        print("Epoch {}, Acc {}, Train Loss {}".format(epoch, accuracy.get(), running_loss/(i+1)))
        evaluate(test_data, ctx, net)

def evaluate(test_data, ctx, net):
    accuracy = 0
    for i, (inputs, seq_lens, token_types, labels) in enumerate(test_data):
        inputs = gluon.utils.split_and_load(inputs, ctx)
        seq_lens = gluon.utils.split_and_load(seq_lens, ctx)
        token_types = gluon.utils.split_and_load(token_types, ctx)
        labels = gluon.utils.split_and_load(labels, ctx)
        for inp, seq_len, token_type, label in zip(inputs, seq_lens, token_types, labels):
            out = net(inp, token_type, seq_len)
            accuracy += (out.argmax(axis=1).squeeze() == label).mean().copyto(mx.cpu()) / len(ctx)
        accuracy.wait_to_read()
    print("Test Acc {}".format(accuracy.asscalar()/(i+1)))

def predict_sentiment(net, ctx, vocabulary, bert_tokenizer, sentence):
    ctx = ctx[0] if isinstance(ctx, list) else ctx
    max_len = 128
    padding_id = vocabulary[vocabulary.padding_token]

    transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_len, pad=False, pair=False)
    dataset = gluon.data.SimpleDataset([[sentence]])
    dataset = dataset.transform(transform)
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=padding_id),
        nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=padding_id))
    predict_data = gluon.data.DataLoader(dataset, batchify_fn=batchify_fn,
                                         batch_size=1)

    for i, (inputs, seq_len, token_types) in enumerate(predict_data):
        inputs = mx.nd.array(inputs).as_in_context(ctx)
        token_types = mx.nd.array(token_types).as_in_context(ctx)
        seq_len = mx.nd.array(seq_len, dtype='float32').as_in_context(ctx)
        out = net(inputs, token_types, seq_len)
        label = nd.argmax(out, axis=1)
        return 'positive' if label.asscalar() == 1 else 'negative'
