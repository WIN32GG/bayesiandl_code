from argparse import ArgumentParser
from bayeformers import to_bayesian
from collections import namedtuple
from tensorboardX import SummaryWriter
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
from queue import Queue
from threading import Thread
from scipy.special import softmax

import bayeformers.nn as bnn
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Report:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total                    : float = 0.0
        self.acc                      : float = 0.0
        self.acc_std                  : float = 0.0
        self.nll                      : float = 0.0
        self.log_prior                : float = 0.0
        self.log_variational_posterior: float = 0.0
        self.em                       : float = 0.0
        self.f1                       : float = 0.0


class Dumper:
    
    DUMPER_CLOSE = "#EOF"
    
    def __init__(self, filename: str = None, append_postfix = False) -> None:
        if not filename:
            filename = f'unnamed.dump'
        self.append_postfix = append_postfix
        self.original_filename : str = filename
        self.data = {}
        self.data_section = False
        self.open()
        self.closing = False

    def __call__(self, name: str = None, value = None):
        if name is None:
            self.data_section = True
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.data_section:
            self.data_section = False
            if len(self.data) > 0:
                # self.queue.put(self.data)
                self.file_handle.write(json.dumps(self.data)+'\n')
                self.data = {}
    
    def close(self):
        self.closing = True
        # self.queue.put(Dumper.DUMPER_CLOSE
    
    def open(self):
        if self.append_postfix:
            postfix = "." + "".join([random.choice(string.ascii_letters + string.digits) for n in range(5)]).upper()
        else:
            postfix = ""
        self.filename = f'{self.original_filename}{postfix}'
        print(f'Dumping results to {self.filename}')
        self.file_handle = open(self.filename, 'w+')


    def __setitem__(self, name: str, value):
        if type(value) in [torch.Tensor, np.ndarray]:
            value = value.tolist()
        self.data[name] = value

    
def to_list(tensor: torch.Tensor) -> List[torch.Tensor]:
    return tensor.detach().cpu().tolist()


def dic2cuda(dic: Dict, device: str) -> Dict:
    for key, value in dic.items():
        if isinstance(value, torch.Tensor):
            dic[key] = value.to(device)

    return dic


def setup_model(model_name: str, lower_case: bool, args) -> Tuple[nn.Module, nn.Module]:
    config    = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    # model.qa_outputs = nn.Sequential(
    #     nn.Linear(args.hidden_size, 1024),
    #     nn.Tanh(),
    #     nn.Linear(1024, 1024),
    #     nn.Tanh(),
    #     nn.Linear(1024, 1024),
    #     nn.Tanh(),
    #     nn.Linear(1024, model.num_labels)
    # )

    return model, tokenizer

def setup_fake_dataset(data_dir: str, postfix: str, **kwargs) -> Tuple[Dataset, torch.Tensor, torch.Tensor]:
    test = False
    cached_path = os.path.join(data_dir, f"fake{postfix}")
    if os.path.isfile(cached_path):
        ckpt = torch.load(cached_path)
        return ckpt["dataset"], ckpt["examples"], ckpt["features"]
    
    raise Error(f'Dataset not found: {cached_path}')

def setup_squadv1_dataset(data_dir: str, postfix: str, test: bool = False, **kwargs) -> Tuple[Dataset, torch.Tensor, torch.Tensor]:
    cached_path = os.path.join(data_dir, f"dev{postfix}")
    if os.path.isfile(cached_path):
        ckpt = torch.load(cached_path)
        return ckpt["dataset"], ckpt["examples"], ckpt["features"]
    
    raise Error(f'Dataset not found: {cached_path}')


def setup_inputs(data: Iterable, model_name: str, model: nn.Module, test: bool = False) -> Dict[str, torch.Tensor]:
    inputs = {
        "input_ids"      : data[0],
        "attention_mask" : data[1],
        "token_type_ids" : data[2],
        "start_positions": data[3] if not test else None,
        "end_positions"  : data[4] if not test else None,
        "feature_indices": None if not test else data[3],
    }

    if test:
        del inputs["start_positions"]
        del inputs["end_positions"]
    else:
        del inputs["feature_indices"]

    if ("xlm" in model_name) or ("roberta" in model_name) or ("distilbert" in model_name) or ("camembert" in model_name):
        del inputs["token_type_ids"]

    return inputs


def softlogmeanexp(inputs: torch.Tensor) -> torch.Tensor: #[SAMPLE, BATCH_SIZE, MAX_SEQ_LENGTH]
    return torch.log(torch.exp(F.softmax(inputs, dim=2)).mean(0))

def sample_bayesian(
    model: bnn.Model, inputs: Dict[str, torch.Tensor], samples: int, batch_size: int, max_seq_len: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start_logits               : torch.Tensor     = torch.zeros(samples, batch_size, max_seq_len).to(device)
    end_logits                 : torch.Tensor     = torch.zeros(samples, batch_size, max_seq_len).to(device)
    log_prior                  : torch.Tensor     = torch.zeros(samples                         ).to(device)
    log_variational_posterior  : torch.Tensor     = torch.zeros(samples                         ).to(device)
   
    dp = type(model) == torch.nn.DataParallel
    for sample in range(samples):
        outputs                           = model(**inputs)
        start_logits[sample]              = outputs[-2]
        end_logits[sample]                = outputs[-1]
        log_prior[sample]                 = model.module.log_prior() if dp else model.log_prior()
        log_variational_posterior[sample] = model.module.log_variational_posterior() if dp else model.log_variational_posterior()

    raw_start_logits          = start_logits
    raw_end_logits            = end_logits
    start_logits              = start_logits.mean(0) #softlogmeanexp(start_logits)
    end_logits                = end_logits.mean(0)   #softlogmeanexp(end_logits)
    log_prior                 = log_prior.mean()
    log_variational_posterior = log_variational_posterior.mean()

    return raw_start_logits, raw_end_logits, start_logits, end_logits, log_prior, log_variational_posterior

# def write_outputs:
#     metadata = {
#     'outputs' : [{
#       'type': 'web-app',
#       'storage': 'gcs',
#       'source': static_html_path,
#     }, {
#       'type': 'web-app',
#       'storage': 'inline',
#       'source': '<h1>Hello, World!</h1>',
#     }]
#   }
#   with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
#     json.dump(metadata, f)

def main(args):
    # EXP: str, 
    MODEL_NAME        = args.model_name
    DEVICE            = args.device
    BATCH_SIZE        = args.batch_size
    SAMPLES           = args.samples
    MAX_SEQ_LENGTH    = args.max_seq_length
    MAX_QUERY_LENGTH  = args.max_query_length
    MAX_ANSWER_LENGTH = args.max_answer_length
    N_BEST_SIZE       = args.n_best_size
    NULL_SCORE_THRESH = args.null_score_thresh
    LOWER_CASE        = args.lower_case
    LOADER_OPTIONS    = { "num_workers": 10, "pin_memory": True }
    DATA_DIR          = args.dataset_dir
    POSTFIX           = args.postfix
    DEVICE_LIST       = list(range(*map(int, args.device_list.split(','))))
    DELTA             = args.delta if args.delta >= 0 else None
    LOGS              = args.logs
    SKIP_N_FIRSTS     = args.skip_n_firsts

    if args.dumper_path and os.path.isfile(args.dumper_path) and args.skip_if_exists:
        print("Skipping")
        exit(0)

    is_baye = args.type == 'bayesian'
    report = Report() 
    os.makedirs(os.path.dirname(args.dumper_path), exist_ok=True)
    dumper = Dumper(args.dumper_path)

    model, tokenizer = setup_model(MODEL_NAME, LOWER_CASE, args)
    if is_baye:
        print('Evaluating a bayesian model')
        model = to_bayesian(model, delta=DELTA, skip_n_firsts=SKIP_N_FIRSTS)
    model = model.to(DEVICE)
    
    model.load_state_dict(torch.load(args.checkpoint_path))
    
    model = torch.nn.DataParallel(model, device_ids=DEVICE_LIST)
    model = model.to(DEVICE)
    # squadv1 = {
    #     "max_seq_length"  : MAX_SEQ_LENGTH,
    #     "doc_stride"      : DOC_STRIDE,
    #     "max_query_length": MAX_QUERY_LENGTH,
    #     "threads"         : THREADS
    # }
    
    test_dataset,  test_examples,  test_features  = setup_squadv1_dataset(DATA_DIR, POSTFIX)
    fake_dataset,  fake_examples,  fake_features  = setup_fake_dataset(DATA_DIR, POSTFIX)

    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, **LOADER_OPTIONS)    

    model.eval()
    report.reset()
    
    with dumper("test"):
        with torch.no_grad():
            results = []
            pbar    = tqdm(test_loader, desc="Test")
            for inputs in pbar:
                inputs          = setup_inputs(inputs, MODEL_NAME, model, True)
                inputs          = dic2cuda(inputs, DEVICE)
                feature_indices = inputs["feature_indices"]
                if is_baye:
                    B               = inputs["input_ids"].size(0)

                del inputs["feature_indices"]

                if is_baye:
                    samples = sample_bayesian(model, inputs, SAMPLES, B, MAX_SEQ_LENGTH, DEVICE)
                    rsl, rel, start_logits, end_logits, log_prior, log_variational_posterior = samples
                    
                    #SHAPE: (SAMPLES, BATCH_SIZE, MAX_SEQ_LENGTH) to (BATCH_SIZE, SAMPLES, MAX_SEQ_LENGTH)
                    rsl = np.moveaxis(rsl.detach().cpu().numpy(), 0, 1)
                    rel = np.moveaxis(rel.detach().cpu().numpy(), 0, 1)
                    
                    start_logits_list = start_logits.tolist()
                    end_logits_list   = end_logits.tolist()
                else:
                    outputs = model(**inputs)

                for i, feature_idx in enumerate(feature_indices):
                    eval_feature             = test_features[feature_idx.item()]
                    fake_feature             = fake_features[feature_idx.item()]
                    unique_id                = int(eval_feature.unique_id)

                    if is_baye:
                        result       = SquadResult(unique_id, start_logits_list[i], end_logits_list[i])
                        results.append(result)
                    else:
                        # output                   = [to_list(out[i]) for out in outputs]
                        start_logits, end_logits = to_list(outputs[-2][i]), to_list(outputs[-1][i])
                        result                   = SquadResult(unique_id, start_logits, end_logits)
                        results.append(result)
                    
                    with dumper():
                        dumper['start_position'] = fake_feature.start_position
                        dumper['end_position']   = fake_feature.end_position
                        dumper['unique_id']      = unique_id
                        if is_baye:
                            dumper['start_logits']  = rsl[i]#start_logits_list[i]
                            dumper['end_logits']    = rel[i]#end_logits_list[i]
                        else:
                            dumper['start_logits']   = start_logits
                            dumper['end_logits']     = end_logits

        predictions = compute_predictions_logits(
            test_examples, test_features, results,
            N_BEST_SIZE, MAX_ANSWER_LENGTH, LOWER_CASE,
            os.path.join(LOGS, f"preds.test.json"),
            os.path.join(LOGS, f"nbestpreds.test.json"),
            None, True, False, NULL_SCORE_THRESH, tokenizer
        )

    dumper.close()
        
    results      = squad_evaluate(test_examples, predictions)
    report.em    = results["exact"]
    report.f1    = results["f1"]
    report.total = results["total"]

    print(f'em={report.em}, f1={report.f1}, total={report.total}')
    # TODO put elsewhere

if __name__ == '__main__':
    parser = ArgumentParser()

    # Data
    parser.add_argument('--skip_if_exists', '-s', type=bool, default=False)
    parser.add_argument('--checkpoint_path', '-l', type=str)
    parser.add_argument('--dataset_dir', '-d', type=str, default="")
    parser.add_argument('--postfix', type=str, default="-v1.1.pth")

    # Model
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--type', type=str, default='frequentist', help="'frequentist'/'bayesian")

    # OutputThe Hurt Locker
    parser.add_argument('--stats_path', type=str, default='./report.json')
    parser.add_argument('--dumper', type=bool, default=False, help='Dump raw results')
    parser.add_argument('--logs', type=str, default="./")
    parser.add_argument('--dumper_path', '-o', type=str, default=None)

    # Devices
    parser.add_argument('--device',                 type=str,   default="cuda:0")
    parser.add_argument('--device_list',            type=str,   default="0,4")

    # Bayesian
    parser.add_argument('--samples', type=int, default=10)
    
    # Process
    parser.add_argument('--num_labels',             type=int,   default=384)
    parser.add_argument('--hidden_size',            type=int,   default=768)
    parser.add_argument('--skip_n_firsts',          type=int,   default=0)
    parser.add_argument('--max_seq_length',         type=int,   default=384)
    parser.add_argument('--max_query_length',       type=int,   default=30)
    parser.add_argument('--n_best_size',            type=int,   default=20)
    parser.add_argument('--batch_size',             type=int,   default=25)
    parser.add_argument('--max_answer_length',      type=int,   default=30)
    parser.add_argument('--lower_case',             type=bool,  default=True)
    parser.add_argument('--null_score_thresh',      type=float, default=0.0)
    parser.add_argument('--delta',                  type=float, default=0.1)
    
    main(parser.parse_args())