from argparse import ArgumentParser
from collections import namedtuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoConfig
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import SquadV1Processor
from transformers import squad_convert_examples_to_features
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.metrics.squad_metrics import squad_evaluate
from transformers.data.processors.squad import SquadResult
from transformers import BertTokenizer
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from torch import Tensor
import random
import json
import copy
import string

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def setup_model(model_name: str, lower_case: bool) -> Tuple[nn.Module, nn.Module]:
    config    = AutoConfig.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    return model, tokenizer

def process_squadv1(data_dir: str, output_dir: str, force: bool, tokenizer: nn.Module, postfix: str, test: bool = False, **kwargs) -> None:
    cached_path = os.path.join(output_dir, f"{'dev' if test else 'train'}{postfix}")
    if os.path.isfile(cached_path) and not force:
        return
    
    processor   = SquadV1Processor()
    fname       = f"{'dev' if test else 'train'}-v1.1.json"
    getter      = processor.get_dev_examples if test else processor.get_train_examples
    examples    = getter(data_dir, fname)
    features, dataset  = squad_convert_examples_to_features(
        examples         = examples,
        tokenizer        = tokenizer,
        is_training      = not test,
        return_dataset   = "pt",
        **kwargs
    )

    os.makedirs(os.path.dirname(cached_path), exist_ok=True)
    torch.save({ "dataset": dataset, "examples": examples, "features": features }, cached_path)


def setup_fake_dataset(data_dir: str, output_dir: str, force: bool, tokenizer: nn.Module, postfix, **kwargs) -> Tuple[Dataset, torch.Tensor, torch.Tensor]:
    cached_path = os.path.join(output_dir, f"fake{postfix}")#os.path.join(data_dir, f"{'dev' if test else 'train'}v1.pth")
    if os.path.isfile(cached_path) and not force:
        return 
    
    processor   = SquadV1Processor()
    fname       = f"dev-v1.1.json"
    getter      = processor.get_train_examples
    examples    = getter(data_dir, fname)
    features, dataset  = squad_convert_examples_to_features(
        examples         = examples,
        tokenizer        = tokenizer,
        is_training      = True,
        return_dataset   = "pt",
        **kwargs
    )

    os.makedirs(os.path.dirname(cached_path), exist_ok=True)
    torch.save({ "dataset": dataset, "examples": examples, "features": features }, cached_path)


def main(args):
    _, tokenizer = setup_model(args.model_name, args.lower_case)

    squadv1 = {
        "max_seq_length"  : args.max_seq_length,
        "doc_stride"      : args.doc_stride,
        "max_query_length": args.max_query_length,
        "threads"         : args.threads
    }

    if args.train:
        process_squadv1(args.raw_dataset_dir, args.output_dir, args.force, tokenizer=tokenizer, postfix=args.postfix, test=False, **squadv1)
    if args.test:
        process_squadv1(args.raw_dataset_dir, args.output_dir, args.force, tokenizer=tokenizer, postfix=args.postfix, test=True,  **squadv1)
    if args.fake:
        setup_fake_dataset(args.raw_dataset_dir, args.output_dir, args.force, tokenizer=tokenizer, postfix=args.postfix, **squadv1)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--raw_dataset_dir', '-d',  type=str,   default='../../dataset/squadv1')
    parser.add_argument('--output_dir', '-o',       type=str,   default='../../dataset/cached/squadv1')
    parser.add_argument('--postfix',                type=str,   default="-v1.1.pth")

    parser.add_argument('--train',                  action="store_true")
    parser.add_argument('--test',                   action="store_true")
    parser.add_argument('--fake',                   action="store_true")
    parser.add_argument('--force',                  action="store_true")
    parser.add_argument('--max_seq_length',         type=int,   default=384)
    parser.add_argument('--doc_stride',             type=int,   default=4096)
    parser.add_argument('--max_query_length',       type=int,   default=64)
    parser.add_argument('--threads',                type=int,   default=4)
    parser.add_argument('--model_name',             type=str,   default='distilbert-base-uncased')
    parser.add_argument('--lower_case',             type=bool,  default=True)

    main(parser.parse_args())
