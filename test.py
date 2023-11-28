#!/bin/python3
from argparse import ArgumentParser
from bayeformers import to_bayesian
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
from scipy.special import softmax

import bayeformers.nn as bnn
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.cuda import amp
import functools


def setup_model(model_name: str, lower_case: bool, hidden_size) -> Tuple[nn.Module, nn.Module]:
    config    = AutoConfig.from_pretrained(model_name)#, hidden_size=hidden_size)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    # model.qa_outputs = nn.Sequential(
    #     nn.Linear(hidden_size, 1024),
    #     nn.Tanh(),
    #     nn.Linear(1024, 1024),
    #     nn.Tanh(),
    #     nn.Linear(1024, 1024),
    #     nn.Tanh(),
    #     nn.Linear(1024, model.num_labels)
    # )

    return model, tokenizer

def make_tensors(inp: Dict):
    for k in inp:
        inp[k] = torch.tensor(inp[k]).unsqueeze(0).cuda()

bayesian = True



if __name__ == '__main__':
    model, tokenizer = setup_model("distilbert-base-uncased", True, 12 * 64)

    if bayesian:
        model = to_bayesian(model, skip_n_firsts = 86, delta = 0.1).cuda()
        print("Model created")
        model.load_state_dict(torch.load("../baye_atester/b_model_trained.pth"))
        print("Model loaded")

        context = "Tesla gained experience in telephony and electrical engineering before emigrating to the United States in 1884 to work for Thomas Edison in New York City. He soon struck out on his own with financial backers, setting up laboratories and companies to develop a range of electrical devices. His patented AC induction motor and transformer were licensed by George Westinghouse, who also hired Tesla for a short time as a consultant. His work in the formative years of electric power development was involved in a corporate alternating current/direct current War of Currents as well as various patent battles."
        question = "What other invention of Tesla's did Westinghouse license?"

        inputs = tokenizer(context, question)
        make_tensors(inputs)
        print("Running model")
        # print(inputs)
        outputs = model(**inputs)

        start = outputs.start_logits.squeeze(0).argmax()
        end   = outputs.end_logits.squeeze(0).argmax()

        print(tokenizer.decode(inputs['input_ids'].squeeze()))
        print(start.item(), end.item())

        print(tokenizer.decode(inputs['input_ids'].squeeze()[start:end]))