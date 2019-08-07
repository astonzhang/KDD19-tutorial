KDD19 Tutorial: From Shallow to Deep Language Representations: Pre-training, Fine-tuning, and Beyond
==================================================================

<h3>Time: Thu, August 08, 2019 - 9:30am - 12:30 pm | 1:00 pm - 4:00 pm</h3>
<h3>Location: Dena’ina Center, Kahtnu 1 & 2-Level 2, 600 W. Seventh Avenue Anchorage, AK 99501</h3>

<span style="color:grey">Presenters: [Aston Zhang](https://www.astonzhang.com/), [Haibin Lin](https://www.linkedin.com/in/linhaibin/), [Leonard Lausen](https://leonard.lausen.nl/), [Sheng Zha](https://www.linkedin.com/in/shengzha/), and [Alex Smola](https://alex.smola.org/)</span><br/>
<span style="color:grey">Other contributors: [Chenguang Wang](https://sites.google.com/site/raychenguangwang/) and [Mu Li](https://www.cs.cmu.edu/~muli/)</span><br/>

Abstract
--------
Natural language processing (NLP) is at the core of the pursuit for artificial intelligence, with deep learning as the main powerhouse of recent advances. Most NLP problems remain unsolved. The compositional nature of language enables us to express complex ideas, but at the same time making it intractable to spoon-feed enough labels to the data-hungry algorithms for all situations. Recent progress on unsupervised language representation techniques brings new hope. In this hands-on tutorial, we walk through these techniques and see how NLP learning can be drastically improved based on pre-training and fine-tuning language representations on unlabelled text. Specifically, we consider shallow representations in word embeddings such as word2vec, fastText, and GloVe, and deep representations with attention mechanisms such as BERT. We demonstrate detailed procedures and best practices on how to pre-train such models and  fine-tune them in downstream NLP tasks as diverse as finding synonyms and analogies, sentiment analysis, question answering, and machine translation. All the hands-on implementations are with Apache (incubating) MXNet and [GluonNLP](http://gluon-nlp.mxnet.io/), and part of the implementations are available on [Dive into Deep Learning](https://www.d2l.ai).


Agenda
------

| Time            | Tutor          | Title                                                                                                                            |
|-----------------|----------------|----------------------------------------------------------------------------------------------------------------------------------|
| 9:30am-10:00am  | Alex Smola     | [Part 1.1: Basics of hands-on deep learning](#part-11-basics-of-hands-on-deep-learning)                                         |
| 10:00am-11:00am | Alex Smola     | [Part 1.2: Neural Networks](#part-12-neural-networks)                                                                           |
| 11:00am-11:10am |                | Coffee break                                                                                                                     |
| 11:10am-11:30am | Aston Zhang    | [Part 2.1: Shallow language representations in word embedding](#part-21-shallow-language-representations-in-word-embedding)     |
| 11:30am-12:30pm | Aston Zhang    | [Part 2.2: Word Embedding Application](#part-22-word-embedding-application)                                                     |
| 12:30pm-1:00pm  |                | Lunch break                                                                                                                      |
| 1:00pm-2:20pm   | Leonard Lausen | [Part 3: Transformer](#part-3-transformer)                                                                                       |
| 2:20pm-2:30pm   |                | Coffee break                                                                                                                     |
| 2:30pm-3:30pm   | Haibin Lin     | [Part 4.1: Deep language representations with Transformer (BERT)](#part-41-deep-language-representations-with-transformer-bert) |
| 3:30pm-4:00pm   | Haibin Lin     | [Part 4.2: BERT Application](#part-42-bert-application)                                                                        |

### Part 1.1: Basics of Hands-on Deep Learning

**Slides**: [[pdf]](01_ndarray_autograd/Part-1.pdf)

**Notebooks**:

1. NDArray: [[ipynb]](01_ndarray_autograd/ndarray.ipynb)
1. Autograd: [[ipynb]](01_ndarray_autograd/autograd.ipynb)


### Part 1.2: Neural Networks

**Notebooks**:

1. Model: [[ipynb]](02_neural_nets/1-model.ipynb)
1. CNN/RNN: [[ipynb]](02_neural_nets/2-cnn-rnn.ipynb)
1. Sequence: [[ipynb]](02_neural_nets/3-sequence.ipynb)
1. RNN with Gluon: [[ipynb]](02_neural_nets/4-rnn-gluon.ipynb)


### Part 2.1: Shallow language representations in word embedding

**Slides**: [[pdf]](03_word_embedding/Part-2.pdf)


### Part 2.2: Word Embedding Application

**Notebooks**: [[ipynb]](04_word_embedding_app/sim-analogy-sentiment-analysis-rnn-cnn.ipynb)


### Part 3: Transformer

**Slides**: [[pdf]](05_transformer/Part-3.pdf)

**Notebooks**: [[ipynb]](05_transformer/transformer.ipynb)


### Part 4.1: Deep language representations with Transformer (BERT)

**Slides**: [[pdf]](06_bert/Part-4.pdf)


### Part 4.2: BERT Application

**Notebooks**: [[ipynb]](07_bert_app/bert.ipynb)


Links
-----

* [Local installation guide](00_setup/install.ipynb)
* [Source code of the `d2l` (v0.10.1) package](d2l-0.10.1.py)
* This tutorial is based on [Dive into Deep Learning](https://www.d2l.ai) and [GluonNLP](http://gluon-nlp.mxnet.io/).
