#!/bin/python3
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
# from transformers.optimization import AdamW
from torch.optim import AdamW, SGD
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

# Very usefull
# from torch import autograd
# autograd.set_detect_anomaly(True)

F32_ε  = 1.1920929e-07

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

def amp_func(f):
    @functools.wraps(f)
    def f_(*args, **kwargs):
        with amp.autocast():
            return f(*args, **kwargs)
    return f_

def to_list(tensor: torch.Tensor) -> List[torch.Tensor]:
    return tensor.detach().cpu().tolist()


def dic2cuda(dic: Dict, device: str) -> Dict:
    for key, value in dic.items():
        if isinstance(value, torch.Tensor):
            dic[key] = value.to(device)

    return dic


def setup_model(model_name: str, lower_case: bool, args: "Namespace") -> Tuple[nn.Module, nn.Module]:
    config    = AutoConfig.from_pretrained(model_name) # hidden_size=args.hidden_size #! removed hidden_size for now
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

def setup_squadv1_dataset(data_dir: str, postfix: str, tokenizer: nn.Module, test: bool = False, **kwargs) -> Tuple[Dataset, torch.Tensor, torch.Tensor]:
    cached_path = os.path.join(data_dir, f"{'dev' if test else 'train'}{postfix}")
    if not os.path.isfile(cached_path):
        raise Exception('Dataset not found')
    ckpt = torch.load(cached_path)
    return ckpt["dataset"], ckpt["examples"], ckpt["features"]


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


def sample_bayesian(
    model: bnn.Model, inputs: Dict[str, torch.Tensor], samples: int, batch_size: int, max_seq_len: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start_logits               : torch.Tensor     = torch.zeros(batch_size, max_seq_len, requires_grad=True).to(device)
    end_logits                 : torch.Tensor     = torch.zeros(batch_size, max_seq_len, requires_grad=True).to(device)
    log_prior                  : torch.Tensor     = torch.zeros(1                      , requires_grad=True).to(device)
    log_variational_posterior  : torch.Tensor     = torch.zeros(1                      , requires_grad=True).to(device)

    for sample in range(samples):
        outputs                           = model(**inputs)
        start_logits                     += outputs[-2]
        end_logits                       += outputs[-1]
        log_prior                         = log_prior                 + model.log_prior()
        log_variational_posterior         = log_variational_posterior + model.log_variational_posterior()

    # raw_start_logits          = start_logits
    # raw_end_logits            = end_logits
    start_logits              = start_logits              / samples
    end_logits                = end_logits                / samples
    log_prior                 = log_prior                 / samples
    log_variational_posterior = log_variational_posterior / samples

    return 0, 0, start_logits, end_logits, log_prior, log_variational_posterior

def load_model(model, args):
    if args.load_checkpoint:
      print(f'Loading checkpoint: {args.load_checkpoint}')
      model.load_state_dict(torch.load(args.load_checkpoint))

# AVUC Utils

H       = lambda p: -1 * torch.sum(p * torch.log(p), axis=1) # BATCH_SIZE, MAX_SEQ_LENGTH
Σ       = lambda e, ids: torch.tensor(0.) if len(ids) == 0 else e[list(ids)].sum()
inter   = lambda a, b  : a.intersection(b)


log     = torch.log

def loss_avuc(p, y, uth=0.5, apply_softmax=True): # probs: for frequentist: direct, bayesian: mean
    """
        probs:  BATCH_SIZE, ELEMS
        labels: BATCH_SIZE
    """

    if apply_softmax:
        p     = F.softmax(p, dim=1)
    u     = H(p)
    p, ŷ  = p.max(axis=1)
    tu    = torch.tanh(u)

    id_accurate   = set([i for i in range(len(y)) if ŷ[i] == y[i]])
    id_inaccurate = set([i for i in range(len(y)) if ŷ[i] != y[i]])
    id_certain    = set([i for i in range(len(y)) if u[i] <= uth])
    id_uncertain  = set([i for i in range(len(y)) if u[i] >  uth])
    
    nau = Σ( p * tu,             inter(id_accurate,   id_uncertain))
    nac = Σ( p * (1 - tu),       inter(id_accurate,   id_certain))
    nic = Σ( (1 - p) * (1 - tu), inter(id_inaccurate, id_certain))
    niu = Σ( (1 - p) * tu,       inter(id_inaccurate, id_uncertain))

    return log( 1 + (( nau + nic ) / (nac + niu + 1)) )

def avuc_calibration(p):
    return H(F.softmax(p, dim=1)).mean()

def get_calibration(bmodel, loader, model_name, device, samples, max_seq_length, fraction = 0.1):
    probabilities_start = []
    probabilities_end   = []

    dataset_elements_numbers = int(len(loader) * fraction)
    print(f'Calibrating on {dataset_elements_numbers} elements')

    with torch.no_grad():
        pbar = tqdm(loader, desc="Calibration")
        for batch_num, inputs in enumerate(pbar):
            if batch_num > dataset_elements_numbers:
                break
            inputs = setup_inputs(inputs, model_name, bmodel)
            inputs = dic2cuda(inputs, device)
            
            start_positions = inputs["start_positions"]
            end_positions   = inputs["end_positions"]

            # if is_baye:
            B = inputs["input_ids"].size(0)
            _, _, start_logits, end_logits, log_prior, log_variational_posterior = sample_bayesian(bmodel, inputs, samples, B, max_seq_length, device)
            # else:
            #     outputs      = model(**inputs)
            #     start_logits = outputs[1]
            #     end_logits   = outputs[2]

            # Setup Outputs
            # prediction = torch.zeros(SAMPLES, img.size(0), 10).cuda()

            # Sample Loop (VI)
            # for s in range(SAMPLES):
            #     prediction[s] = torch.softmax(start_logits, 1)
            
            probabilities_start.append(start_logits.detach().cpu().tolist())
            probabilities_end.append(end_logits.detach().cpu().tolist())

            # prediction = prediction.mean(0).detach().cpu().numpy().tolist()
            # for p in prediction:
            #     probabilities.append(p.tolist())
    cs = torch.tensor(probabilities_start)
    ce = torch.tensor(probabilities_end)

    cs = cs.reshape(cs.size(0) * cs.size(1), -1) # Should now be (BATCH_SIZE, MAX_SEQ)
    ce = ce.reshape(ce.size(0) * ce.size(1), -1)

    cs = avuc_calibration(cs)
    ce = avuc_calibration(ce)

    return cs.item(), ce.item()

def main(args):
    EXP               = args.exp
    MODEL_NAME        = args.model_name
    DELTA             = args.delta if args.delta >= 0 else None
    WEIGHT_DECAY      = args.weight_decay
    DEVICE            = args.device
    DEVICE_LIST       = list(range(*map(int, args.device_list.split(','))))
    EPOCHS            = args.epochs
    BATCH_SIZE        = args.batch_size
    SAMPLES           = args.samples
    FREEZE            = args.freeze
    SKIP_N_FIRSTS     = args.skip_n_firsts
    LOGS              = args.logs
    DOC_STRIDE        = args.doc_stride
    MAX_SEQ_LENGTH    = args.max_seq_length
    MAX_QUERY_LENGTH  = args.max_query_length
    MAX_ANSWER_LENGTH = args.max_answer_length
    N_BEST_SIZE       = args.n_best_size
    NULL_SCORE_THRESH = args.null_score_thresh
    LOWER_CASE        = args.lower_case
    THREADS           = args.threads
    LOADER_OPTIONS    = { "num_workers": args.num_workers, "pin_memory": True }
    LR                = args.lr
    ADAM_EPSILON      = args.adam_epsilon
    N_WARMUP_STEPS    = args.n_warmup_steps
    MAX_GRAD_NORM     = args.max_grad_norm
    DATA_DIR          = args.dataset_dir
    POSTFIX           = args.postfix
    CALIBRATION_FRACTION = args.calibration_fraction

    # Loss scalers
    α       = float(args.alpha)     # ELBO
    β       = float(args.beta)      # AVUC
    γ       = float(args.gamma)     # NLL


    if args.save_checkpoint and os.path.isfile(args.save_checkpoint) and args.skip_if_exists:
        print("Skipping as output already exists")
        exit(0)

    writer : SummaryWriter  = SummaryWriter(os.path.join(LOGS, "data"))
    print("Tensorboard ready")

    scaler = amp.GradScaler()

    model, tokenizer = setup_model(MODEL_NAME, LOWER_CASE, args)
    model = model.to(DEVICE)

    squadv1 = {
        "max_seq_length"  : MAX_SEQ_LENGTH,
        "doc_stride"      : DOC_STRIDE,
        "max_query_length": MAX_QUERY_LENGTH,
        "threads"         : THREADS
    }
    
    train_dataset, train_examples, train_features = setup_squadv1_dataset(DATA_DIR, POSTFIX, tokenizer=tokenizer, test=False, **squadv1)
    test_dataset,  test_examples,  test_features  = setup_squadv1_dataset(DATA_DIR, POSTFIX, tokenizer=tokenizer, test=True,  **squadv1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  **LOADER_OPTIONS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, **LOADER_OPTIONS)

    is_baye = args.type == "bayesian" 
    
    if args.transform_loaded:
        load_model(model, args)
    
    model.train() 
    
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    if is_baye:
        print("Using Bayesian Model")
        # bayesian head freeze is controled by the FREEZE parameter
        model = to_bayesian(model, delta = DELTA, freeze = FREEZE, skip_n_firsts = SKIP_N_FIRSTS) # to_bayesian does a deep copy, gc will handle collection

    model = model.to(DEVICE)
    
    if not args.transform_loaded:
        load_model(model, args)

    # Disable data parallel for now
    # model = torch.nn.DataParallel(model, device_ids=DEVICE_LIST)
    model = model.to(DEVICE)
    # model.forward = amp_func(model.forward) # AMP force
    
    # decay           = [param for name, param in model.named_parameters() if name     in ["bias", "LayerNorm.weight"]]
    # no_decay        = [param for name, param in model.named_parameters() if name not in ["bias", "LayerNorm.weight"]]
    # params_decay    = { "params": decay,    "weight_decay": WEIGHT_DECAY }
    # params_no_decay = { "params": no_decay, "weight_decay": 0.0 }
    # parameters      = [params_decay, params_no_decay]

    if is_baye:
        print(f'The model has {len(model.bayesian_children)} bayesian modules')

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optim     = SGD(model.parameters(), lr=LR, momentum=.9, nesterov=True) # AdamW(model.parameters(), lr=LR, eps=ADAM_EPSILON) #
    scheduler = get_linear_schedule_with_warmup(optim, N_WARMUP_STEPS, EPOCHS)
    report    = Report()

    for epoch in tqdm(range(EPOCHS), desc="Epoch"):
        report.reset()

        if is_baye:
            uth_start, uth_end = get_calibration(model, iter(train_loader), MODEL_NAME, DEVICE, SAMPLES, MAX_SEQ_LENGTH, CALIBRATION_FRACTION)
            print("Calibration ", uth_start, uth_end)

        pbar = tqdm(train_loader, desc="Train")
        for batch_num, inputs in enumerate(pbar):
            inputs = setup_inputs(inputs, MODEL_NAME, model)
            inputs = dic2cuda(inputs, DEVICE)
            
            start_positions = inputs["start_positions"]
            end_positions   = inputs["end_positions"]

            optim.zero_grad()

            # with amp.autocast():
            # with torch.autograd.detect_anomaly():
            if is_baye:
                B = inputs["input_ids"].size(0)
                samples = sample_bayesian(model, inputs, SAMPLES, B, MAX_SEQ_LENGTH, DEVICE)
                _, _, start_logits, end_logits, log_prior, log_variational_posterior = samples
            else:
                outputs      = model(**inputs)
                start_logits = outputs[1]
                end_logits   = outputs[2]
                
            ignored_idx            = start_logits.size(1)
            start_logits           = start_logits.clamp_(0, ignored_idx)
            end_logits             = end_logits.clamp_(0, ignored_idx)
            criterion.ignore_index = ignored_idx

            start_loss = criterion(start_logits, start_positions)
            end_loss   = criterion(  end_logits,   end_positions)

            start_acc  = (torch.argmax(start_logits, dim=1) == start_positions).float().sum()
            end_acc    = (torch.argmax(  end_logits, dim=1) ==   end_positions).float().sum()

            𝓛nll     = 0.5 * (start_loss + end_loss)
            acc      = 0.5 * (start_acc  + end_acc)

            if is_baye:
                # start_acc_std = np.std([(torch.argmax(start_logits.clamp(0, ignored_idx), dim=1) == start_positions).float().sum().item() for start_logits in raw_start_logits])
                # end_acc_std   = np.std([(torch.argmax(  end_logits.clamp(0, ignored_idx), dim=1) ==   end_positions).float().sum().item() for   end_logits in raw_end_logits])
                # acc_std       = 0.5 * (start_acc_std + end_acc_std)
                
                𝓛elbo         = (log_variational_posterior - log_prior) / len(train_loader)
                𝓛avuc         = loss_avuc(start_logits, start_positions, uth_start) + loss_avuc(end_logits, end_positions, uth_end)
                loss          = α * 𝓛elbo + β * 𝓛avuc + γ * 𝓛nll # Total
                
            else:
                loss    = 𝓛nll
                
            #scaler.scale(loss).backward() 
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            #scaler.step(optim) 
            optim.step()
            #scaler.update()

            report.total += loss.item()      / len(train_loader)
            report.acc   += acc.item() * 100 / len(train_dataset)

            # if is_baye:
            #     report.𝓛nll                      += 𝓛nll.item()                      / len(train_loader)
            #     report.log_prior                 += log_prior.item()                 / len(train_loader)
            #     report.log_variational_posterior += log_variational_posterior.item() / len(train_loader)
            #     report.acc_std                   += acc_std                          / len(train_loader)
            
            nll_item  = 𝓛nll.item()
            loss_item = loss.item()

            if is_baye:
                elbo_item = 𝓛elbo.item()
                avuc_item = 𝓛avuc.item()
                
                step =  len(train_loader) * epoch + batch_num

                # writer.add_histogram('qa_output_element', bmodel)

                writer.add_scalar("log_prior", log_prior.item(), step)
                writer.add_scalar("log_variational_posterior", log_variational_posterior.item(), step)
                writer.add_scalar("𝓛elbo",  elbo_item, step)
                writer.add_scalar("𝓛avuc",  avuc_item, step)
                writer.add_scalar("𝓛nll",   nll_item,  step)
                writer.add_scalar("𝓛total", loss_item, step)

                pbar.set_postfix(nll=nll_item, acc=report.acc, 𝓛elbo=elbo_item, 𝓛avuc=avuc_item)
            else:
                pbar.set_postfix(nll=nll_item, acc=report.acc)
    
        scheduler.step()
        # writer.add_scalar("train_nll", report.total, epoch)
        # writer.add_scalar("train_acc"sample_bayesian, report.acc,   epoch)
    
    if args.save_checkpoint:
        print("Saving to ", args.save_checkpoint)
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        torch.save(model.module.state_dict() if type(model) == torch.nn.DataParallel else model.state_dict(), args.save_checkpoint)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Pipelines
    parser.add_argument('--skip_if_exists', '-s',   type=bool,  default=False)

    # Dataset
    parser.add_argument('--dataset_dir', '-d',      type=str,   default='../../dataset/squadv1')
    parser.add_argument('--postfix',                type=str,   default="-v1.1.pth")

    # Train type 
    parser.add_argument('--type',                   type=str,   default='frequentist', help="Either 'frequentist' or 'bayesian', not both at once")
    parser.add_argument('--epochs',                 type=int,   default=3)

    # Bayesian Specific
    parser.add_argument('--freeze',                 action="store_true")
    parser.add_argument('--delta',                  type=float, default=10e-1)
    parser.add_argument('--samples',                type=int,   default=10)
    parser.add_argument('--skip_n_firsts',          type=int,   default=0)

    # Checkpoints
    parser.add_argument('--load_checkpoint',  '-l', type=str,   default=None)
    parser.add_argument('--save_checkpoint',  '-o', type=str,   default=None)
    parser.add_argument('--transform_loaded', '-t', type=bool,  default=False)

    # Logs
    parser.add_argument('--logs',                   type=str,   default="/logs")
    parser.add_argument('--report_stats',           type=bool,  default=True)
    parser.add_argument('--exp',                    type=str,   default='exp')

    # Devices
    parser.add_argument('--device',                 type=str,   default="cuda:0")
    parser.add_argument('--device_list',            type=str,   default="0,4")

    parser.add_argument('--alpha',                  type=float, default=1.0)
    parser.add_argument('--beta',                   type=float, default=3.0)
    parser.add_argument('--gamma',                  type=float, default=1.0)
    

    # Model, Trainer & Loader
    parser.add_argument('--num_labels',             type=int,   default=384)
    parser.add_argument('--hidden_size',            type=int,   default=768)
    parser.add_argument('--max_seq_length',         type=int,   default=384)
    parser.add_argument('--max_answer_length',      type=float, default=30)
    parser.add_argument('--doc_stride',             type=int,   default=4096)
    parser.add_argument('--max_query_length',       type=int,   default=64)
    parser.add_argument('--threads',                type=int,   default=4)
    parser.add_argument('--model_name',             type=str,   default='distilbert-base-uncased')
    parser.add_argument('--lower_case',             type=bool,  default=True)
    parser.add_argument('--batch_size',             type=int,   default=25)
    parser.add_argument('--n_best_size',            type=int,   default=20)
    parser.add_argument('--null_score_thresh',      type=float, default=0.0)
    parser.add_argument('--lr',                     type=float, default=1e-3)
    parser.add_argument('--adam_epsilon',           type=float, default=1e-8)
    parser.add_argument('--weight_decay',           type=float, default=1e-2)
    parser.add_argument('--n_warmup_steps',         type=int,   default=0)
    parser.add_argument('--max_grad_norm',          type=int,   default=100)
    parser.add_argument('--calibration_fraction',   type=float, default=0.1)
    parser.add_argument('--num_workers',            type=int,   default=10)

    arguments = parser.parse_args()
    print(arguments)
    main(arguments)

