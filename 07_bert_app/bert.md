# BERT Pre-training and Fine-tuning

In this notebook, you will understand how to implement the BERT model for pre-training, and to fine-tune a pre-trained BERT model for sentiment analysis.


## Preparation

First, let's import necessary modules.

Note that utils.py includes some Blocks defined in the previous transformer notebook

```{.python .input}
import random, math

import d2l
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
import gluonnlp as nlp

from utils import PositionalEncoding, MultiHeadAttention 
from utils import AddNorm, PositionWiseFFN, EncoderBlock
from utils import train_loop, predict_sentiment
```

### Encoder

Different from the transformer encoder, the BERT encoder has an additional embedding for segment information.

<img src="transformer-bert.png" alt="architecture" width="450"/>


### Segment Embedding

![segment embedding](bert-embed.png)

Similar to the Transformer encoder defined in the previous section, the BERT encoder has embeddings for words and positions. The `EncoderBlock` contains position-wise feed-forward network and self-attention blocks to encode inputs. For BERT, the newly added segment embedding captures the segment information of the input sentence pairs, used for the next sentence prediction task.

### BERT Encoder Definition

```{.python .input}
class BERTEncoder(gluon.nn.Block):
    def __init__(self, vocab_size, units, hidden_size,
                 num_heads, num_layers, dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        # segment_embed for segment information
        self.segment_embed = gluon.nn.Embedding(2, units)
        self.word_embed = gluon.nn.Embedding(vocab_size, units)
        self.pos_encoding = PositionalEncoding(units, dropout)
        self.blks = gluon.nn.Sequential()
        for i in range(num_layers):
            self.blks.add(EncoderBlock(units, hidden_size, num_heads, dropout))

    def forward(self, words, segments, mask, *args):
        X = self.word_embed(words) + self.segment_embed(segments)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X, mask)
        return X
```

### Using BERT Encoder

Now let's test the BERTEncoder with a data batch of 2 sentence pairs, each with 8 words. Random integers are used to represent words for demonstration purpose. For segment information, we use 0 to indicate the word comes from the first sentence, 1 to indicate the second setence.

```{.python .input}
encoder = BERTEncoder(vocab_size=30000, units=768, hidden_size=3072,
                      num_heads=12, num_layers=12, dropout=0.1)
encoder.initialize()

num_samples, num_words = 2, 8
# random words for testing
words = nd.random.randint(low=0, high=30000, shape=(num_samples, num_words))
# the corresponding segment information for each word
segments = nd.array([[0,0,0,0,1,1,1,1],[0,0,0,1,1,1,1,1]])

encodings = encoder(words, segments, None)
print(encodings.shape) # (batch_size, num_words, units)
```

### Next Sentence Classifier


Let us take a look at the first pre-training task: next sentence prediction. For this task, the encoding of the first token (the "[CLS]" token) is passed to a feed-forward network to make prediction.

```{.python .input}
class NSClassifier(gluon.nn.Block):
    def __init__(self, units=768, **kwargs):
        super(NSClassifier, self).__init__(**kwargs)
        self.classifier = gluon.nn.Sequential()
        self.classifier.add(gluon.nn.Dense(units=units, flatten=False, activation='tanh'))
        # binary classification layer
        self.classifier.add(gluon.nn.Dense(units=1, flatten=False))

    def forward(self, X, *args):
        X = X[:, 0, :]  # get the encoding of the first token
        pred = self.classifier(X)
        return pred
```

### Using Next Sentence Classifier

Since next sentence prediction is a binary classification problem, we can use `SigmoidBinaryCrossEntropyLoss` as the loss function. In the following code block, we pass the encoding results to the `NSClassifier` to get the next sentence prediction. We use 1 as the label for true next sentence, and 0 otherwise. The prediction result and the label are then passed to the loss function for loss evaluation.

```{.python .input}
ns_classifier = NSClassifier()
ns_classifier.initialize()

ns_pred = ns_classifier(encodings) # (batch_size, 1)
ns_label = nd.array([0, 1]) # 1 for true next setence, 0 otherwise
ns_loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss()
ns_loss = ns_loss_fn(ns_pred, ns_label).mean()
print(ns_pred.shape, ns_loss.shape)
```

### Masked Language Model Decoder

Masked language modeling is one of the two pre-training task, where random positions are masked and the model needs to reconstruct the masked words. In the masked language model decoder, we first use `gather_nd` to pick the dense vectors representing words at masked position. Then a feed-forward network is applied on them, followed by a fully-connected layer to predict the unnormalized score for all words in the vocabulary.

```{.python .input}
class MLMDecoder(gluon.nn.Block):
    def __init__(self, vocab_size, units, **kwargs):
        super(MLMDecoder, self).__init__(**kwargs)
        self.decoder = gluon.nn.Sequential()
        self.decoder.add(gluon.nn.Dense(units, flatten=False))
        # Gaussian Error Linear Units as the activation function [4]
        self.decoder.add(gluon.nn.GELU())
        self.decoder.add(gluon.nn.LayerNorm())
        # classification layer for `vocab_size` classes
        self.decoder.add(gluon.nn.Dense(vocab_size, flatten=False))

    def forward(self, X, masked_positions, *args):
         # gather encodings at mask positions
        X = nd.gather_nd(X, masked_positions)
        pred = self.decoder(X)
        return pred
```

### Using Masked Language Model Decoder

In the following code block, we pass the encoding results to the `MLMDecoder` to get the masked language model prediction. We generate some random word indices as the label for demonstration purpose. For multi-class classification, we can use `SoftmaxCrossEntropyLoss` as the loss function. The prediction result and the label are then passed to the loss function for loss evaluation.

```{.python .input}
decoder = MLMDecoder(vocab_size=30000, units=768)
decoder.initialize()

mlm_positions = nd.array([[0,1],[4,8]])
mlm_label = nd.array([100, 200])
mlm_pred = decoder(encodings, mlm_positions) # (batch_size, vocab_size)
mlm_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_loss = mlm_loss_fn(mlm_pred, mlm_label).mean()
print(mlm_pred.shape, mlm_loss.shape)
```

## BERT Fine-tuning (Sentiment Analysis)

In this section, we fine-tune the BERT Base model for sentiment analysis on the IMDB dataset.


### BERT for Sentence Classification

Let's first take
a look at the BERT model
architecture for single sentence classification below:
<div style="width:
500px;">![bert-sa](bert-sa.png)</div>

Here the model takes a sentences and pools the representation of the first token in the sequence.
Note that the original BERT model was trained for a masked language model and next-sentence prediction tasks, which includes layers for language model decoding and
classification. These layers will not be used for fine-tuning sentence classification.

### Get Pre-train BERT Model

We can load the pre-trained BERT fairly easily using the model API in GluonNLP, which returns the vocabulary along with the model. We include the pooler layer of the pre-trained model by setting `use_pooler` to `True`.
The list of pre-trained BERT models available in GluonNLP can be found [here](../../model_zoo/bert/index.rst).

```{.python .input}
ctx = mx.gpu(0) if mx.test_utils.list_gpus() else mx.cpu()
bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                            dataset_name='book_corpus_wiki_en_uncased',
                                            pretrained=True, ctx=ctx,
                                            use_decoder=False, use_classifier=False)
print(bert_base)
```

### Model for Fine-tuning

Now that we have loaded the BERT model, we only need to attach an additional layer for classification.
The `BERTClassifier` class uses a BERT base model to encode sentence representation, followed by a `nn.Dense` layer for classification.

```{.python .input}
class BERTClassifier(gluon.nn.Block):
    def __init__(self, bert, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        # extra layer used for classification
        self.classifier = gluon.nn.Dense(num_classes)

    def forward(self, inputs, segment_types, seq_len):
        seq_encoding, cls_encoding = self.bert(inputs, segment_types, seq_len)
        return self.classifier(cls_encoding)
```

### Model Initialization and Loss

We only need to initialize the classification layer. The encoding layers are already initialized with pre-trained weights.

```{.python .input}
net = BERTClassifier(bert_base, 2)
net.classifier.initialize(ctx=ctx)
loss_fn = gluon.loss.SoftmaxCELoss()
```

## Data Preprocessing for BERT

For this tutorial, we need to do a bit of preprocessing before feeding our data introduced
the BERT model. 

### Loading the dataset

We again use the IMDB dataset, but for this time, downloading using the GluonNLP data API.

```{.python .input}
train_dataset_raw = nlp.data.IMDB('train')
test_dataset_raw = nlp.data.IMDB('test')
```

We then use the transform API to transform the raw scores to positive labels and negative labels.

```{.python .input}
def transform_label(data):
    # Transform label into position / negative
    text, label = data
    return text, 1 if label >= 5 else 0
train_dataset = train_dataset_raw.transform(transform_label)
test_dataset = test_dataset_raw.transform(transform_label)
```

### BERT-specific Transformations

To use the pre-trained BERT model, we need to tokenize the data in the same
way it was trained. We need to perform the following transformations:
- tokenize the inputs into word pieces
- insert [CLS] at the beginning of a sentence
- insert [SEP] at the end of a sentence
- generate segment ids

### BERT Vocabulary

Let's take a look at the vocabulary previously downloaded from the `model.get_model` API:

```{.python .input}
print(vocabulary)
print('index for [CLS] = ', vocabulary['[CLS]'])
print('index for [SEP] = ', vocabulary['[SEP]'])
```

### Tokenization for BERT

We can perform tokenization with `data.BERTTokenizer` API. The vocabulary from pre-trained model is used to construct the tokenizer. Let's take a look at the tokenization result of a short sample:

```{.python .input}
tokenizer = nlp.data.BERTTokenizer(vocabulary)
text, label = train_dataset[16854]
print('original text:')
print(text)
print('\ntokenized text:')
print(' '.join(tokenizer(text)))
```

To process sentences with BERT-style '[CLS]', '[SEP]' tokens, you can use `data.BERTSentenceTransform` API.

```{.python .input}
def transform_fn(text, label):
    max_len = 128
    transform = nlp.data.BERTSentenceTransform(tokenizer, max_len,
                                               pad=False, pair=False)
    data, length, segment_type = transform([text])
    data = data.astype('float32')
    length = length.astype('float32')
    segment_type = segment_type.astype('float32')
    return data, length, segment_type, label
```

```{.python .input}
data, length, segment_type, label = transform_fn(*train_dataset[0])
print('words = ', data.astype('int32'))
print('segments = ', segment_type.astype('int32'))
```

### Batchify and Data Loader

```{.python .input}
padding_id = vocabulary[vocabulary.padding_token]
batchify_fn = nlp.data.batchify.Tuple(
        # words: the first dimension is the batch dimension
        nlp.data.batchify.Pad(axis=0, pad_val=padding_id),
        # valid length
        nlp.data.batchify.Stack(),
        # segment type : the first dimension is the batch dimension
        nlp.data.batchify.Pad(axis=0, pad_val=padding_id),
        # label
        nlp.data.batchify.Stack(np.float32))
```

```{.python .input}
batch_size = 32
train_data = gluon.data.DataLoader(train_dataset.transform(transform_fn, lazy=False),
                                   batchify_fn=batchify_fn, shuffle=True,
                                   batch_size=batch_size)
test_data = gluon.data.DataLoader(test_dataset.transform(transform_fn, lazy=False), 
                                  batchify_fn=batchify_fn,
                                  shuffle=False, batch_size=batch_size)
```

### Training Loop

Now we have all the pieces to put together, and we can finally start fine-tuning the
model with a few epochs.

```{.python .input}
num_epoch = 3
lr = 0.00005
train_loop(net, train_data, test_data, num_epoch, lr, ctx, loss_fn)
```

### Prediction

```{.python .input}
predict_sentiment(net, ctx, vocabulary, tokenizer, 'this movie is so great')
```

## Conclusion

In this tutorial, we showed how to fine-tune sentiment analysis model with pre-trained BERT parameters. In GluonNLP, this can be done with such few, simple steps. All we did was apply a BERT-style data transformation to pre-process the data, automatically download the pre-trained model, and feed the transformed data into the model, all within 50 lines of code!

For more fine-tuning scripts, visit the [BERT model zoo webpage](http://gluon-nlp.mxnet.io/model_zoo/bert/index.html).

## References

[1] Devlin, Jacob, et al. "Bert:
Pre-training of deep
bidirectional transformers for language understanding."
arXiv preprint
arXiv:1810.04805 (2018).

[2] Dolan, William B., and Chris
Brockett.
"Automatically constructing a corpus of sentential paraphrases."
Proceedings of
the Third International Workshop on Paraphrasing (IWP2005). 2005.

[3] Peters,
Matthew E., et al. "Deep contextualized word representations." arXiv
preprint
arXiv:1802.05365 (2018).

[4] Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." arXiv preprint arXiv:1606.08415 (2016).

For fine-tuning, we only need to initialize the last classifier layer from scratch. The other layers are already initialized from the pre-trained model weights.
