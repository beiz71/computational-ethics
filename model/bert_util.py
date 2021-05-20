from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

import torch.autograd as autograd


class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(MyBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, note=""):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.note = note


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid


class MAProcessor(object):
    
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "clean_train_set.pkl")), "clean") + \
               self._create_examples(self._read_tsv(os.path.join(data_dir, "dirty_train_set.pkl")), "dirty")

    def get_test_examples(self, data_dir):
        clean_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "clean_test_set.pkl")), "clean")
        dirty_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "dirty_test_set.pkl")), "dirty")
        return clean_test_ex, dirty_test_ex
    
    # def get_conde_examples(self, data_dir):
    #     return self._create_examples(self._read_tsv(os.path.join(data_dir, "conde_set.pkl")), "dirty")
    #
    # def get_sarca_examples(self, data_dir):
    #     return self._create_examples(self._read_tsv(os.path.join(data_dir, "sarca_set.pkl")), "dirty")
    #
    # def get_hostile_examples(self, data_dir):
    #     return self._create_examples(self._read_tsv(os.path.join(data_dir, "hostile_set.pkl")), "dirty")

    # correct predicted examples
    def get_correct_conde_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "correct_conde_set.pkl")), "dirty")

    def get_correct_sarca_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "correct_sarca_set.pkl")), "dirty")

    def get_correct_hostile_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "correct_hostile_set.pkl")), "dirty")

    # wrong predicted examples
    def get_wrong_conde_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "wrong_conde_set.pkl")), "dirty")

    def get_wrong_sarca_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "wrong_sarca_set.pkl")), "dirty")

    def get_wrong_hostile_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "wrong_hostile_set.pkl")), "dirty")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line[0]
            if set_type == "clean":
                label = "0"
            elif set_type == "dirty":
                label = "1"
            else:
                raise ValueError("Check your set type")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "rb") as f:
            pairs = pickle.load(f)
            return pairs


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid=example.guid))
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


def accuracy(out, label_ids):
    # axis-0: seqs in batch; axis-1: potential labels of seq
    outputs = np.argmax(out, axis=1)
    matched = outputs == label_ids
    num_correct = np.sum(matched)
    num_total = len(label_ids)
    return num_correct, num_total
