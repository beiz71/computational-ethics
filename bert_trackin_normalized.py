from __future__ import absolute_import, division, print_function

import argparse
import datetime
import json
import logging
import os
import random
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

import torch.autograd as autograd

from model.bert_util import *


def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_trained_epoch",
                        default=0,
                        type=int,
                        required=True,
                        help="Number of epochs the trained model went through.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--trained_model_dir",
                        default="",
                        type=str,
                        help="Where is the fine-tuned (with the cloze-style LM objective) BERT model?")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--freeze_bert',
                        action='store_true',
                        help="Whether to freeze BERT")
    parser.add_argument('--full_bert',
                        action='store_true',
                        help="Whether to use full BERT")
    parser.add_argument('--num_train_samples',
                        type=int,
                        default=-1,
                        help="-1 for full train set, otherwise please specify")
    parser.add_argument('--test_idx',
                        type=int,
                        default=1,
                        help="test index we want to examine")
    parser.add_argument('--start_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument('--end_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument("--probe_type",
                        default="",
                        type=str,
                        help="Select probing example type: conde, sarca, hostile")
    parser.add_argument("--do_correct",
                        action='store_true',
                        help="Whether to probe from correct prediction.")
    parser.add_argument("--do_wrong",
                        action='store_true',
                        help="Whether to probe from wrong prediction.")

    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_seed(args.seed)

    if args.do_correct and args.do_wrong:
        raise ValueError("Only one of `do_correct` and `do_wrong` can be set True at the same time.")

    if not os.path.exists(args.output_dir):  # bert_tracin_normalized_output
        os.makedirs(args.output_dir)

    folder_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.do_correct:
        folder_name += "-correct"
    if args.do_wrong:
        folder_name += "-wrong"

    args.output_dir = os.path.join(args.output_dir, folder_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "params.json"), "w") as fw:
        json.dump(args.__dict__, fw, indent=2)

    args.device = device

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        filename="%s/train.log" % args.output_dir,
                        filemode='w')
    logging.info("***** Start Logging *****")

    # prepare data processor
    ma_processor = MAProcessor()
    label_list = ma_processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # get train dataloader
    train_examples = ma_processor.get_train_examples(args.data_dir)
    train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
    logging.info("***** Train set *****")
    logging.info("  Num examples = %d", len(train_examples))
    print(f"Num examples = {len(train_examples)}")
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id)
    train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=1)

    # get probe dataloader
    test_examples = None
    if args.do_correct:
        if args.probe_type == "conde":
            test_examples = ma_processor.get_correct_conde_examples(args.data_dir)
        elif args.probe_type == "sarca":
            test_examples = ma_processor.get_correct_sarca_examples(args.data_dir)
        elif args.probe_type == "hostile":
            test_examples = ma_processor.get_correct_hostile_examples(args.data_dir)
        else:
            raise ValueError("Check your probe type")

    if args.do_wrong:
        if args.probe_type == "conde":
            test_examples = ma_processor.get_wrong_conde_examples(args.data_dir)
        elif args.probe_type == "sarca":
            test_examples = ma_processor.get_wrong_sarca_examples(args.data_dir)
        elif args.probe_type == "hostile":
            test_examples = ma_processor.get_wrong_hostile_examples(args.data_dir)
        else:
            raise ValueError("Check your probe type")

    test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
    logging.info("***** Test set *****")
    logging.info("  Num examples = %d", len(test_examples))
    print(f"Num examples = {len(test_examples)}")
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=1)

    # start probing
    logging.info("***** Start Probing *****")
    agg_influence_dict = dict()  # aggregates influence score from checkpoints
    
    for epoch_idx in range(args.num_trained_epoch):
        # Load model checkpoint
        model = MyBertForSequenceClassification.from_pretrained(os.path.join(args.trained_model_dir, f"epoch_{epoch_idx}"), num_labels=num_labels)
        model.to(device)
        param_optimizer = list(model.named_parameters())
        # print(model.__dict__())
        # print(param_optimizer[:5])
        if args.freeze_bert:
            frozen = ['bert']
        elif args.full_bert:
            frozen = []
        else:
            frozen = ['bert.embeddings.',
                      'bert.encoder.layer.0.',
                      'bert.encoder.layer.1.',
                      'bert.encoder.layer.2.',
                      'bert.encoder.layer.3.',
                      'bert.encoder.layer.4.',
                      'bert.encoder.layer.5.',
                      'bert.encoder.layer.6.',
                      'bert.encoder.layer.7.']  # *** change here to filter out params we don't want to track ***
        param_influence = []
        for n, p in param_optimizer:
            if not any(fr in n for fr in frozen):
                param_influence.append(p)
            elif 'bert.embeddings.word_embeddings.' in n:
                pass  # need gradients through embedding layer for computing saliency map
            else:
                p.requires_grad = False
        param_shape_tensor = []
        param_size = 0
        for p in param_influence:
            tmp_p = p.clone().detach()
            param_shape_tensor.append(tmp_p)
            param_size += torch.numel(tmp_p)
        logging.info("  Parameter size = %d", param_size)
        print(f"Parameter size = {param_size}")
    
        # Calculate influence
        influence_dict = dict()
        ihvp_dict = dict()

        for tmp_idx, (input_ids, input_mask, segment_ids, label_ids) in enumerate(test_dataloader):
            if args.start_test_idx != -1 and args.end_test_idx != -1:
                if tmp_idx < args.start_test_idx:
                    continue
                if tmp_idx > args.end_test_idx:
                    break
            else:
                if tmp_idx < args.test_idx:
                    continue
                if tmp_idx > args.test_idx:
                    break

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            influence_dict[tmp_idx] = np.zeros(len(train_examples))

            ######## L_TEST GRADIENT ########
            model.eval()
            model.zero_grad()
            test_loss = model(input_ids, segment_ids, input_mask, label_ids)
            test_grads = autograd.grad(test_loss, param_influence)
            ################

            ihvp_dict[tmp_idx] = gather_flat_grad(test_grads).detach().cpu()  # put to CPU to save GPU memory

        for tmp_idx in ihvp_dict.keys():
            ihvp_dict[tmp_idx] = ihvp_dict[tmp_idx].to(args.device)

        set_seed(args.seed)

        for train_idx, (_input_ids, _input_mask, _segment_ids, _label_ids) in enumerate(tqdm(train_dataloader, desc="Train set index")):
            model.eval()
            _input_ids = _input_ids.to(device)
            _input_mask = _input_mask.to(device)
            _segment_ids = _segment_ids.to(device)
            _label_ids = _label_ids.to(device)

            ######## L_TRAIN GRADIENT ########
            model.zero_grad()
            train_loss = model(_input_ids, _segment_ids, _input_mask, _label_ids)
            train_grads = autograd.grad(train_loss, param_influence)
            ################

            with torch.no_grad():
                for tmp_idx in ihvp_dict.keys():
                    influence_dict[tmp_idx][train_idx] = F.cosine_similarity(ihvp_dict[tmp_idx], gather_flat_grad(train_grads), dim=0).item()

        for k, v in influence_dict.items():
            if k not in agg_influence_dict:
                agg_influence_dict[k] = v
            else:
                agg_influence_dict[k] = agg_influence_dict[k] + v
                
    # save results
    for k, v in agg_influence_dict.items():
        influence_filename = f"influence_test_idx_{k}.pkl"
        pickle.dump(v, open(os.path.join(args.output_dir, influence_filename), "wb"))


if __name__ == "__main__":
    main()
