KDD19 Tutorial: From Shallow to Deep Language Representations: Pre-training, Fine-tuning, and Beyond
==================================================================

<h3>Time: Thu, August 08, 2019 - 9:30am - 12:30 pm | 1:00 pm - 4:00 pm</h3>
<h3>Location: Dena’ina Center, 600 W. Seventh Avenue Anchorage, AK 99501</h3>

<span style="color:grey">Presenter: Aston Zhang, Haibin Lin, Leonard Lausen, Sheng Zha, Alex Smola</span><br/>


Abstract
--------
Natural language processing (NLP) is at the core of the pursuit for artificial intelligence, with deep learning as the main powerhouse of recent advances. Most NLP problems remain unsolved. The compositional nature of language enables us to express complex ideas, but at the same time making it intractable to spoon-feed enough labels to the data-hungry algorithms for all situations. Recent progress on unsupervised language representation techniques brings new hope. In this hands-on tutorial, we walk through these techniques and see how NLP learning can be drastically improved based on pre-training and fine-tuning language representations on unlabelled text. Specifically, we consider shallow representations in word embeddings such as word2vec, fastText, and GloVe, and deep representations with attention mechanisms such as BERT. We demonstrate detailed procedures and best practices on how to pre-train such models and  ne-tune them in downstream NLP tasks as diverse as finding synonyms and analogies, sentiment analysis, question answering, and machine translation. All the hands-on implementations are with Apache (incubating) MXNet and [GluonNLP](http://gluon-nlp.mxnet.io/), and part of the implementations are available on [Dive into Deep Learning](www.d2l.ai).


Target Audience
--------
We are targeting engineers, scientists, and instructors in the field of natural language processing, data mining, text mining, deep learning, machine learning, and artificial intelligence. While the audience with a good background in these areas would benefit the most from this tutorial, it will give general audience and newcomers an introductory pointer to the presented materials.


Agenda
------

| Time        | Tutor        | Title                                                                  | Slides    | Notebooks  |
|-------------|-------------|------------------------------------------------------------------------|-----------|------------|
| 9:30am-10:00am | Alex Smola | Basics of hands-on deep learning                                             |  | [ndarray](https://github.com/astonzhang/KDD19-tutorial/blob/master/01_basics/ndarray.ipynb), [autograd](https://github.com/astonzhang/KDD19-tutorial/blob/master/01_basics/autograd.ipynb) |
| 10:00am-10:50am | Alex Smola | Neural networks                                    |  |  MLP, CNN, RNN |
| 11:00am-11:45am | Aston Zhang  | Shallow language representations in word embedding                           |           | [word2vec](https://github.com/astonzhang/KDD19-tutorial/blob/master/02_word_embedding/word2vec.ipynb), [fasttext](https://github.com/astonzhang/KDD19-tutorial/blob/master/02_word_embedding/fasttext.ipynb), [GloVe](https://github.com/astonzhang/KDD19-tutorial/blob/master/02_word_embedding/glove.ipynb), [pre-train](https://github.com/astonzhang/KDD19-tutorial/blob/master/02_word_embedding/word2vec-gluon.ipynb) |
| 11:45am-12:30pm | Aston Zhang   | Application                                       | | [analogy](https://github.com/astonzhang/KDD19-tutorial/blob/master/03_finetuning_word_embedding/similarity-analogy.ipynb), [sa-rnn](https://github.com/astonzhang/KDD19-tutorial/blob/master/03_finetuning_word_embedding/sentiment-analysis-rnn.ipynb), [sa-cnn](https://github.com/astonzhang/KDD19-tutorial/blob/master/03_finetuning_word_embedding/sentiment-analysis-cnn.ipynb)|
| 12:30pm-1:00pm |    | Lunch break                                                         | |  |
| 1:00pm-2:20pm | Leonard Lausen   | Transfomer                                                         | |  |
| 2:30pm-3:30pm | Haibin Lin   | Deep language representations with Transfomer (BERT)                                 |  |   |
| 3:30pm-4:00pm | Haibin Lin   | Application                                                 |           | [finetune](https://github.com/astonzhang/KDD19-tutorial/blob/master/06_bert/bert.ipynb) |
