import csv
import logging
import os
import sys
from io import open
from tqdm import tqdm
import numpy as np
import pandas
from collections import Counter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize 
# nltk.download('stopwords')
words = stopwords.words('english')
stop_words = set(stopwords.words('english')) 

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class HotpotQAProcessor(object):
    def get_examples(self, input_file):
        logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            pandas.read_csv(input_file), set_type=input_file)

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, df, set_type):
        examples = []
        for (i, row) in df.iterrows():
            guid = "%s-%s" % (set_type, i)
            text_a = row['Sub_question']
            text_a = text_a.replace('[Answer of Question0]','')
            text_b = '{} {}'.format(row['Title'],row['Sentences'])
            label = row['Label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,tokenizer, verbose=False):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    sent_sub_query_pairs = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        sub_query = example.text_a.lower()#问题的子问题
        word_tokens = word_tokenize(sub_query) 
        filtered_sentence = [] 
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w) 
        sub_query = " ".join(filtered_sentence)#去除停用词
        sent = example.text_b#文章的句子
        tokens_sub_query = tokenizer.tokenize(sub_query)#分词
        tokens_sentence = tokenizer.tokenize(sent.lower())
        _truncate_seq_pair(tokens_sub_query, tokens_sentence, max_seq_length - 3)
                
        # Feature ids
        tokens = ["[CLS]"] + tokens_sub_query + ["[SEP]"] + tokens_sentence + ["[SEP]"]#将问题短语和文章中所有的句子放在一起送进bert中计算匹配度
        sent_sub_query_pairs.append(tokens)
        segment_ids = [0] * (len(tokens_sub_query) + 2) + [1] * (len(tokens_sentence) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
                
        # Mask and Paddings
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        title = example.text_b
        if ex_index < 5 and verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    #return (preds == labels).mean()
    outputs = np.argmax(preds, axis=1)
    return np.sum(outputs == labels)


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "hotpotqa":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)



TASK_NUM_LABELS = {
    "hotpotqa": 2
}


