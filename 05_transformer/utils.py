# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utilities for transformer."""

import numpy as np
import math
import mxnet as mx
from mxnet import nd
import time
import logging
import io
import nmt
import hyperparameters as hparams
import inspect

from matplotlib import pyplot as plt
import IPython
import pygments

def setup_source():
    IPython.display.display(
        IPython.display.HTML("<style>{pygments_css}</style>".format(
            pygments_css=pygments.formatters.HtmlFormatter().get_style_defs(
                '.highlight'))))


def source(function):
    code = inspect.getsource(function)
    html = pygments.highlight(
        code, pygments.lexers.PythonLexer(),
        pygments.formatters.HtmlFormatter(style='colorful'))
    IPython.core.display.display(IPython.core.display.HTML(html))

def use_svg_display():
    """Use the svg format to display plot in jupyter."""
    IPython.display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Change the default figure size"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """A utility function to set matplotlib axes"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=None,
         figsize=(3.5, 2.5), axes=None):
    """Plot multiple lines"""
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    if isinstance(X, nd.NDArray): X = X.asnumpy()
    if isinstance(Y, nd.NDArray): Y = Y.asnumpy()
    if not hasattr(X[0], "__len__"): X = [X]
    if Y is None: X, Y = [[]]*len(X), X
    if not hasattr(Y[0], "__len__"): Y = [Y]
    if len(X) != len(Y): X = X * len(Y)
    if not fmts: fmts = ['-']*len(X)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if isinstance(x, nd.NDArray): x = x.asnumpy()
        if isinstance(y, nd.NDArray): y = y.asnumpy()
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def evaluate(model, data_loader, test_loss_function, translator, tgt_vocab, detokenizer, context):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(data_loader):
        src_seq = src_seq.as_in_context(context)
        tgt_seq = tgt_seq.as_in_context(context)
        src_valid_length = src_valid_length.as_in_context(context)
        tgt_valid_length = tgt_valid_length.as_in_context(context)
        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = test_loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)
        # Translate
        samples, _, sample_valid_length = \
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = detokenizer(nmt.bleu._bpe_to_words(sentence),
                                                return_str=True)
    return avg_loss, real_translation_out

def translate(translator, src_seq, src_vocab, tgt_vocab, detokenizer, ctx):
    src_sentence = src_vocab[src_seq.split()]
    src_sentence.append(src_vocab[src_vocab.eos_token])
    src_npy = np.array(src_sentence, dtype=np.int32)
    src_nd = mx.nd.array(src_npy)
    src_nd = src_nd.reshape((1, -1)).as_in_context(ctx)
    src_valid_length = mx.nd.array([src_nd.shape[1]]).as_in_context(ctx)
    samples, _, sample_valid_length = \
        translator.translate(src_seq=src_nd, src_valid_length=src_valid_length)
    max_score_sample = samples[:, 0, :].asnumpy()
    
    sample_valid_length = sample_valid_length[:, 0].asnumpy()
    translation_out = []
    for i in range(max_score_sample.shape[0]):
        translation_out.append(
            [tgt_vocab.idx_to_token[ele] for ele in
             max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    real_translation_out = [None for _ in range(len(translation_out))]
    for ind, sentence in enumerate(translation_out):
        real_translation_out[ind] = detokenizer(nmt.bleu._bpe_to_words(sentence),
                                                return_str=True)
    return real_translation_out              
                
def train_one_epoch(epoch_id, model, train_data_loader, trainer, label_smoothing, loss_function, grad_interval, average_param_dict, update_average_param_dict, step_num, ctx):
    log_avg_loss = 0
    log_wc = 0
    loss_denom = 0
    step_loss = 0
    log_start_time = time.time()
    for batch_id, seqs in enumerate(train_data_loader):
        if batch_id % grad_interval == 0:
            step_num += 1
            new_lr = hparams.lr / math.sqrt(hparams.num_units) * min(1. / math.sqrt(step_num), step_num * hparams.warmup_steps ** (-1.5))
            trainer.set_learning_rate(new_lr)
        src_wc, tgt_wc, bs = np.sum([(shard[2].sum(), shard[3].sum(), shard[0].shape[0])
                                     for shard in seqs], axis=0)
        src_wc = src_wc.asscalar()
        tgt_wc = tgt_wc.asscalar()
        loss_denom += tgt_wc - bs
        seqs = [[seq.as_in_context(context) for seq in shard]
                for context, shard in zip([ctx], seqs)]
        Ls = []
        with mx.autograd.record():
            for src_seq, tgt_seq, src_valid_length, tgt_valid_length in seqs:
                out, _ = model(src_seq, tgt_seq[:, :-1],
                               src_valid_length, tgt_valid_length - 1)
                smoothed_label = label_smoothing(tgt_seq[:, 1:])
                ls = loss_function(out, smoothed_label, tgt_valid_length - 1).sum()
                Ls.append((ls * (tgt_seq.shape[1] - 1)) / hparams.batch_size / 100.0)
        for L in Ls:
            L.backward()
        if batch_id % grad_interval == grad_interval - 1 or\
                batch_id == len(train_data_loader) - 1:
            if update_average_param_dict:
                for k, v in model.collect_params().items():
                    average_param_dict[k] = v.data(ctx).copy()
                update_average_param_dict = False
                    
            trainer.step(float(loss_denom) / hparams.batch_size / 100.0)
            param_dict = model.collect_params()
            param_dict.zero_grad()
            if step_num > hparams.average_start:
                alpha = 1. / max(1, step_num - hparams.average_start)
                for name, average_param in average_param_dict.items():
                    average_param[:] += alpha * (param_dict[name].data(ctx) - average_param)
        step_loss += sum([L.asscalar() for L in Ls])
        if batch_id % grad_interval == grad_interval - 1 or\
                batch_id == len(train_data_loader) - 1:
            log_avg_loss += step_loss / loss_denom * hparams.batch_size * 100.0
            loss_denom = 0
            step_loss = 0
        log_wc += src_wc + tgt_wc
        if (batch_id + 1) % (hparams.log_interval * grad_interval) == 0:
            wps = log_wc / (time.time() - log_start_time)
            logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, '
                         'throughput={:.2f}K wps, wc={:.2f}K'
                         .format(epoch_id, batch_id + 1, len(train_data_loader),
                                 log_avg_loss / hparams.log_interval,
                                 np.exp(log_avg_loss / hparams.log_interval),
                                 wps / 1000, log_wc / 1000))
            log_start_time = time.time()
            log_avg_loss = 0
            log_wc = 0