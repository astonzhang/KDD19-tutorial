import logging
import collections
import mxnet as mx
import gluonnlp as nlp
import bert
from mxnet.gluon.model_zoo import model_store

def download_qa_ckpt():
    model_store._model_sha1['bert_qa'] = '7eb11865ecac2a412457a7c8312d37a1456af7fc'
    result = model_store.get_model_file('bert_qa', root='./temp')
    print('Downloaded checkpoint to {}'.format(result))
    return result

def predict(dataset, all_results, vocab):
    tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=True)
    transform = bert.data.qa.SQuADTransform(tokenizer, is_pad=False, is_training=False, do_lookup=False)
    dev_dataset = dataset.transform(transform._transform)
    from bert.bert_qa_evaluate import PredResult, predict

    all_results_np = collections.defaultdict(list)
    for example_ids, pred_start, pred_end in all_results:
        batch_size = example_ids.shape[0]
        example_ids = example_ids.asnumpy().tolist()
        pred_start = pred_start.reshape(batch_size, -1).asnumpy()
        pred_end = pred_end.reshape(batch_size, -1).asnumpy()

        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            all_results_np[example_id].append(PredResult(start=start, end=end))

    all_predictions = collections.OrderedDict()
    top_results = []
    for features in dev_dataset:
        results = all_results_np[features[0].example_id]
    
        prediction, nbest = predict(
            features=features,
            results=results,
            tokenizer=nlp.data.BERTBasicTokenizer(lower=True))
    
        curr_result = {}
        question = features[0].input_ids.index('[SEP]')
        curr_result['context'] = features[0].doc_tokens
        curr_result['question'] = features[0].input_ids[1:question]
        curr_result['prediction'] = nbest[0]
        top_results.append(curr_result)
    return top_results
