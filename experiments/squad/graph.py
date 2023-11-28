#!/bin/python3
import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from scipy.special import softmax
from tqdm import tqdm
from argparse import ArgumentParser

# from bayeformers import to_bayesian
# from collections import namedtuple
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
from transformers import AutoConfig
from transformers import AutoTokenizer
# from transformers import AutoModelForQuestionAnswering
# from transformers import SquadV1Processor
# from transformers import squad_convert_examples_to_features
# from transformers.data.metrics.squad_metrics import squad_evaluate
# from transformers.data.processors.squad import SquadResult
# from transformers.optimization import AdamW
# from transformers.optimization import get_linear_schedule_with_warmup
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.metrics.squad_metrics import squad_evaluate, get_raw_scores
from tqdm import tqdm
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from torch import Tensor

import random
import json
import string
import sys
import scipy.stats

# import bayeformers.nn as bnn
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.data.processors.squad import SquadResult

"""
Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
"Obtaining well calibrated probabilities using bayesian binning."
Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/
"""
from uncertainty_metrics.numpy.general_calibration_error import ece


FEATURE_START = 1000000000

class GraphInputData:

    def __init__(self, filename, label, data_type):
        self.filename = filename
        self.label = label
        assert data_type in ['f', 'b']
        self.is_baye = data_type == 'b'
        self.data = None
        self.check_sanity()

    def check_sanity(self):
        if not os.access(self.filename, os.R_OK):
            raise Exception("Cannot access data file")

# Utils Functions

# Load dataset
def load_dataset(args):
    model_name = args.model_name
    cached_path = os.path.join(args.dataset_dir, 'dev' + args.postfix)

    print(f'Loading tokenizer for {model_name}')
    config    = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    # model     = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    print('Loading dataset', cached_path)
    ckpt = torch.load(cached_path)
    return ckpt["dataset"], ckpt["examples"], ckpt["features"], tokenizer

def entropy(probs: np.ndarray, axis = 0) -> float:
    return -1 * np.sum(probs * np.log(probs), axis=axis)

def cross_entropy(gt_position: int, classes: int, prediction_logits: np.ndarray, eps : float = sys.float_info.epsilon) -> float:
    gt_encoded = np.array([1 if i == gt_position else 0 for i in range(classes)])
    soft = softmax(prediction_logits)
    entropy =  gt_encoded * np.log(soft + eps) + (1 - gt_encoded)*np.log(1 - soft + eps)
    return (-1/classes) * entropy.sum()

def f_score(expected_tokens : np.ndarray, predicted_tokens : np.ndarray):
    correct_tokens = np.intersect1d(expected_tokens, predicted_tokens)

    if len(correct_tokens) == 0:
        return 0
    
    precision      = len(correct_tokens) / len(predicted_tokens)
    recall         = len(correct_tokens) / len(expected_tokens)

    return 2 * (precision * recall) / (precision + recall)

# Bayesian: Get logits from ensemble
def bayesian_get_logits(samples : np.ndarray):
    return samples.mean(0)

# Uncertainty (logits entropy ?, mean of std over samples ? )
def frequentist_uncertainty(start_logits : np.ndarray, end_logits : np.ndarray, **args):
    return entropy(softmax(start_logits)) + entropy(softmax(end_logits))

# Baye sub-functions
def distrib_std(sample_distrib):
    return sample_distrib.std(axis=0).sum()

def entropy_mean(sample_distrib):
    return entropy(sample_distrib, axis=1).mean()

def predictive_uncertainty(sample_distrib):
    return entropy(sample_distrib.mean(axis = 0)) - entropy(sample_distrib, axis=1).mean()

# Main bayesian uncertainty estimator, using given aggregator
def bayesian_uncertainty(start_samples : np.ndarray, end_samples : np.ndarray, bayesian_uncertainty_aggregator, **args):
    ssp = softmax(start_samples, axis=1)
    esp = softmax(end_samples,   axis=1)
    return bayesian_uncertainty_aggregator(ssp) + bayesian_uncertainty_aggregator(esp)

# Prec units
def exact_match(gt_position: int, classes: int, prediction_logits: np.ndarray) -> float:
    return prediction_logits.argmax() == gt_position

# Precision (EM, F1, NLL)
def em_precision(start_logits: np.ndarray, end_logits: np.ndarray, start_position: int, end_position: int, feature: "SquadFeature", args):
    return exact_match(start_position, args.max_seq_length, start_logits) + exact_match(end_position,   args.max_seq_length, end_logits)

def nll_precision(start_logits: np.ndarray, end_logits: np.ndarray, start_position: int, end_position: int, feature: "SquadFeature", args):
    return cross_entropy(start_position, args.max_seq_length, start_logits) + cross_entropy(end_position,   args.max_seq_length, end_logits)

def f_precision(start_logits: np.ndarray, end_logits: np.ndarray, start_position: int, end_position: int, feature: "SquadFeature", args):
    predicted_start     = start_logits.argmax() # argmax of logits is same as argmx of softmax
    predicted_end       = end_logits.argmax() 

    if predicted_end < predicted_start:
        return 0.0

    expected_tokens     = feature.input_ids[start_position:end_position   + 1]
    predicted_tokens    = feature.input_ids[predicted_start:predicted_end + 1]

    return f_score(expected_tokens, predicted_tokens)

# Global 
def compute_precision(data, examples, features, tokenizer, precision_estimator, is_baye, args):
    print("Compute precision")
   
    N_BEST_SIZE = 20
    MAX_ANSWER_LENGTH = 30
    LOWER_CASE = True
    NULL_SCORE_THRESH = 0.0

    results = []

    features_id_to_example_id = {f.unique_id: f.qas_id for f in features}

    for elem in tqdm(data):
        if is_baye:
            start_logits = bayesian_get_logits(elem['start_logits'])
            end_logits   = bayesian_get_logits(elem['end_logits'])
        else:
            start_logits = elem['start_logits']
            end_logits   = elem['end_logits']

        results.append(SquadResult(int(elem['unique_id']), start_logits, end_logits))
        
    predictions = compute_predictions_logits(
        examples, features, results,
        N_BEST_SIZE, MAX_ANSWER_LENGTH, LOWER_CASE,
        None, None, None, 
        True, False, NULL_SCORE_THRESH, tokenizer
    )
    _, fscores = get_raw_scores(examples, predictions)

    print("Put back f score")
    for elem in tqdm(data):
        elem['precision'] = fscores[features_id_to_example_id[elem['unique_id']]]
        # elem['precision'] = precision_estimator(start_logits, end_logits, elem['start_position'], elem['end_position'], features[elem['unique_id'] - FEATURE_START], args)
    

    # results = squad_evaluate(examples, predictions)
    # print(results)
    # pdb.set_trace()
    # exit(0)

def compute_uncertainty(data, features, uncertainty_estimator, args, key, bayesian_uncertainty_aggregator = distrib_std):
    print("Compute uncertainty")
    for elem in tqdm(data):
        elem[key] = uncertainty_estimator(elem['start_logits'], elem['end_logits'], bayesian_uncertainty_aggregator = bayesian_uncertainty_aggregator)

def n_th_percentile(elements, n):
    return elements[int(len(elements) * n / 100) + 1:]

def total_prec(elements):
    return np.array(elements).mean()

def total_uncertain(elements):
    return np.array(elements).mean()
        
# Graphs functions

def expected_calibration_error(graph_save_path : str, *inputs : GraphInputData, **args):
    for k,i in enumerate(inputs):

        predictions_start, labels_start = [], []
        predictions_end  , labels_end   = [], []
        
        for elem in i.data:
            if i.is_baye:
                predictions_start.append(softmax(bayesian_get_logits(elem['start_logits'])))
                labels_start.append(elem['start_position'])

                predictions_end.append(softmax(bayesian_get_logits(elem['end_logits'])))
                labels_end.append(elem['end_position'])
            else:
                predictions_start.append(softmax(elem['start_logits']))
                labels_start.append(elem['start_position'])

                predictions_end.append(softmax(elem['end_logits']))
                labels_end.append(elem['end_position'])

        ece_total = ece(labels_start, predictions_start) + ece(labels_end, predictions_end)
        plt.text(0, k/10, f'ECE({i.label}) = {ece_total}')
        print(i.label, ece_total)
    plt.title("Expected Calibration Error")
    plt.savefig(graph_save_path)

def uncertainty_with_precision(graph_save_path : str, *inputs : GraphInputData, **args):
    for inp in inputs:
        uncertainty_fct_precision = [i['uncertainty'] for i in inp.sorted_precision_dta]
        plt.plot([total_uncertain(n_th_percentile(uncertainty_fct_precision, i)) for i in tqdm(range(100))])
    labels = [i.label for i in inputs]
    
    plt.legend(labels=labels)
    plt.title("Uncertainty: Evolution with precision")
    plt.xlabel("Top n-percent most accurate")
    plt.ylabel("Uncertainty level")
    plt.savefig(graph_save_path)

def precision_with_uncertainty(graph_save_path : str, *inputs : GraphInputData, **args):
    for inp in inputs:
        precision_fct_uncertainty = [i['precision'] for i in inp.sorted_uncertainty_dta]
        plt.plot([total_prec(n_th_percentile(precision_fct_uncertainty, i)) for i in tqdm(range(100))])
    labels = [i.label for i in inputs]
    
    plt.legend(labels=labels)
    plt.title("Precision: Evolution with uncertainty")
    plt.xlabel("Top n-percent most certain retained")
    plt.ylabel("Precision")
    plt.savefig(graph_save_path)
    
def uncertainty_repartition(graph_save_path : str, *inputs : GraphInputData, **args):

    labels = [i.label for i in inputs]

    plt.xlabel("Entropy-ordered samples")
    plt.ylabel("Uncertainty level")

    for i in inputs:
        plt.plot(i.sorted_uncertainty)
    
    plt.legend(labels=labels)
    plt.title("Entropy repartition")
    plt.savefig(graph_save_path)


def threshold_repartition(graph_save_path, *inouts: GraphInputData, **args):
    """
    Accuracy on examples with uncertainty > threshold
    """
    pass

## Show examples


valid_hex = '0123456789ABCDEF'.__contains__
def cleanhex(data):
    return ''.join(filter(valid_hex, data.upper()))

def back_fromhex(text, hexcode):
    """print in a hex defined color"""
    hexint = int(cleanhex(hexcode), 16)
    print("\x1B[48;2;{};{};{}m{} \x1B[0m".format(hexint>>16, hexint>>8&0xFF, hexint&0xFF, text), end='')

def print_example(data, features, examples, is_baye):
    feature = features[data['unique_id'] - FEATURE_START]
    example = examples[feature.example_index]
    mapping = feature.token_to_orig_map
    question = example.question_text
    paragraph = example.doc_tokens

    answer_start = softmax(data['start_logits'].mean(0) if is_baye else data['start_logits']).argmax()
    answer_end   = softmax(data['end_logits'].mean(0) if is_baye else data['end_logits']).argmax()

    answer_start = mapping[answer_start] if answer_start in mapping else 0
    answer_end   = mapping[answer_end]   if answer_end   in mapping else 0

    print("================= E X A M P L E =================")
    print('prec=', data['precision'], 'uncertainty=', data['uncertainty'])
    print('???: ', question)
    print('ANS:', example.answer_text)
    for i, tok in enumerate(paragraph):
        c = 'F' if i >= answer_start and i <= answer_end else '0'
        back_fromhex(tok, f'#{c}{c}0000')
    print(" ")

def show_examples(graph_save_path: str, *inputs : GraphInputData, **args):
    features = args['features']
    examples = args['examples']

    for i in inputs:
        print('- - - - ', i.label, ' - - - -')
        print_example(i.sorted_precision_dta[0], features, examples, i.is_baye)    # Example with lowest precision
        print_example(i.sorted_precision_dta[len(i.sorted_precision_dta)//2], features, examples, i.is_baye)    # Example with lowest precision
        print_example(i.sorted_precision_dta[-1],  features, examples, i.is_baye)   # example hightest precision
        print_example(i.sorted_uncertainty_dta[0],  features, examples, i.is_baye)  # example with lowest uncertainty
        print_example(i.sorted_uncertainty_dta[len(i.sorted_uncertainty_dta)//2],  features, examples, i.is_baye)  # example with lowest uncertainty
        print_example(i.sorted_uncertainty_dta[-1],  features, examples, i.is_baye) # example with highest uncertainty

def show_corr(graph_save_path: str, *inputs : GraphInputData, **args):
    corr = {}
    for i in inputs:
        prec = [a['precision'] for a in i.sorted_uncertainty_dta]
        corr[i.label] = scipy.stats.spearmanr(i.sorted_uncertainty, prec)
        # if i.is_baye:
        #     prec = [a['precision'] for a in i.sorted_agg_uncertainty_dta]
        #     corr[i.label+'-agg'] = scipy.stats.spearmanr(i.sorted_agg_uncertainty, prec)

    print(corr)

# End graphs function

def compute_common_data(input : GraphInputData, examples, features, tokenizer, precision_estimator, args):
    bayesian_uncertainty_aggregator = UNCERTAINTY_METRICS[args.bayesian_uncertainty_aggregator]
    compute_precision(input.data, examples, features, tokenizer, precision_estimator, input.is_baye, args)
    compute_uncertainty(input.data, features, bayesian_uncertainty if input.is_baye else frequentist_uncertainty, args, 'uncertainty', bayesian_uncertainty_aggregator)
    # if input.is_baye:
    #     compute_uncertainty(input.data, features, bayesian_uncertainty, args, 'agg_uncertainty', entropy_mean)
    
def compute_sorted_data(input : GraphInputData):
    input.sorted_precision_dta  = sorted(input.data, key=lambda elem: elem['precision'],   reverse = False)
    input.sorted_precision      = [i['precision'] for i in input.sorted_precision_dta]
    
    input.sorted_uncertainty_dta = sorted(input.data, key=lambda elem: elem['uncertainty'], reverse = True)
    input.sorted_uncertainty     = [i['uncertainty'] for i in input.sorted_uncertainty_dta]

    # if input.is_baye:
    #     input.sorted_agg_uncertainty_dta = sorted(input.data, key=lambda elem: elem['agg_uncertainty'], reverse = True)
    #     input.sorted_agg_uncertainty     = [i['agg_uncertainty'] for i in input.sorted_agg_uncertainty_dta]


def load_file(input : GraphInputData):
    current = []
    filename = input.filename
    print("Loading", filename)
    with tqdm(total=os.path.getsize(filename)) as pbar:
        with open(filename) as fh:
            for line in fh:
                pbar.update(len(line.encode('utf-8')))
                enc = json.loads(line)
                enc['start_logits'] = np.array(enc['start_logits'])
                enc['end_logits']   = np.array(enc['end_logits'])
                current.append(enc)
                
    input.data = current

GRAPHS = {
    'uncertainty_repartition': uncertainty_repartition,
    'precision_with_uncertainty': precision_with_uncertainty,
    'uncertainty_with_precision': uncertainty_with_precision,
    'expected_calibration_error': expected_calibration_error,
    'show_examples': show_examples,
    'show_corr': show_corr
}

UNCERTAINTY_METRICS = {
    'distrib_std': distrib_std,
    'entropy_mean': entropy_mean,
    'predictive_uncertainty': predictive_uncertainty # predictive entropy,
}

def main(args):
    data = []

    # Prepare input
    for source in args.source:
        data.append(GraphInputData(*(" ".join(source)).split(',')))
    
    if not args.graph in GRAPHS:
        raise Exception("Unknown graph function")
    
    dataset, examples, features, tokenizer = load_dataset(args)
    graph = GRAPHS[args.graph]

    print("Input Sanity Check passed")
    
    # Load files TODO: para
    for d in data:
        load_file(d)
        compute_common_data(d, examples, features, tokenizer, f_precision, args)
        compute_sorted_data(d)
        
#         r = d.sorted_precision_dta[-1]
#         f = features[d.sorted_precision_dta[-1]['unique_id'] - FEATURE_START]
#         print(f.tokens)
#         print(f.tokens[r['start_position']:r['end_position'] + 1])
#         print(f.tokens[r['start_logits'].mean(0).argmax():r['end_logits'].mean(0).argmax() + 1])

    print("Running Graph", args.graph)
    graph(args.output, *data, dataset=dataset, examples=examples, features=features)

if __name__ == "__main__":
    parser = ArgumentParser()

    """
    pyton graph.py -d ../dataset/processed/squadv1 -o graph.jpg -g uncertainty_repartition  -s ./dump.freq.dump,Frequentist Baseline,f -s ./dump.baye_untrained.dump,Bayesian Baseline,b
    pyton graph.py -d ../dataset/processed/squadv1 -o graph.jpg -g precision_by_uncertainty -s ./dump.freq.dump,Frequentist Baseline,f -s ./dump.baye_untrained.dump,Bayesian Baseline,b
    """

    # Data
    parser.add_argument('--dataset_dir', '-d', type=str, default="")
    parser.add_argument('--postfix', type=str, default="-v1.1.pth")

    # Grapher
    parser.add_argument('--source', '-s', type=str, action='append', nargs='*')
    parser.add_argument('--graph', '-g', type=str)
    parser.add_argument('--output', '-o', type=str, default="graph.jpg")

    # Model
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--type', type=str, default='frequentist', help="'frequentist'/'bayesian")

    # Output
    parser.add_argument('--stats_path', type=str, default='./report.json')
    parser.add_argument('--dumper', type=bool, default=False, help='Dump raw results')
    parser.add_argument('--logs', type=str, default="./")
    parser.add_argument('--dumper_path', type=str, default=None)

    # Devices
    parser.add_argument('--device',                 type=str,   default="cuda:0")
    parser.add_argument('--device_count',           type=int,   default=3)

    # Bayesian
    parser.add_argument('--bayesian_uncertainty_aggregator', type=str, default='distrib_std')
    # parser.add_argument('--samples', type=int, default=10)

    
    # Process
    parser.add_argument('--max_seq_length',         type=int,   default=384)
    # parser.add_argument('--max_query_length',      type=int, default=30)
    # parser.add_argument('--n_best_size', type=int, default=20)
    # parser.add_argument('--batch_size', type=int, default=25)
    # parser.add_argument('--max_answer_length', type=int, default=30)
    parser.add_argument('--lower_case', type=bool, default=True)
    
    main(parser.parse_args())