# -*- coding: utf-8 -*-
"""bayeformers.__init__

The files provides the to_bayesian conversion method.
"""
from bayeformers.nn import TORCH2BAYE
from bayeformers.nn import DROPOUTS
from bayeformers.nn.model import Model
from bayeformers.nn.parameters.initializations import DEFAULT_UNIFORM
from bayeformers.nn.parameters.initializations import Initialization
from bayeformers.nn.parameters.base import Parameter
from bayeformers.nn.parameters.gaussian import DEFAULT_SCALED_GAUSSIAN_MIXTURE
from copy import deepcopy
from typing import Optional

import torch
import torch.nn
from torch.nn import Dropout


def to_bayesian(
    model: nn.Model,
    initialization: Optional[Initialization] = DEFAULT_UNIFORM,
    prior: Optional[Parameter] = DEFAULT_SCALED_GAUSSIAN_MIXTURE,
    delta: float = None, freeze: bool = False, skip_n_firsts: int = 0,
    auto_disable_dropout: bool = True
) -> Model:
    """To Bayesian
    
    Entrypoint of Bayeformers, call to_bayesian on a PyTorch model to
    replace in-place the available swappable layers with their bayesian
    equivalent.

    Arguments:
        model (nn.Model): model to process
        
    Keyword Arguments:
        initialization (Optional[Initialization]): initialization callback
            for the bayesian layers
        prior (Optional[Parameter]): the prior parameters
        delta (float): is the model pretrained? If it not None, the model
            weights will be used to initialize the bayesian weights and delta
            will be used for MOPED posterior init {default: None}
            pretrained loading following:
                    "Specifying Weight Priors in Bayesian Deep Neural Networks
                    with Empirical Bayes" from Krishnan et al.
            reference: https://arxiv.org/pdf/1906.05323.pdf
        freeze (bool): freeze weight's mu if delta is not None {default: False}
        skip_n_firsts (int): Skip the n first layers of the model and 
            only converts those after. n is determined by a model traversal 
            using Model.named_children() {default: 0}
        auto_disable_dropout (bool): Call layer.eval() on all Dropout layers 
            encountered during transformation. Please note that calling 
            model.train() or eval() on the generated model when this option is True
            will have side effects: train will reactivate the Dropouts. Eval
            will deactivate the BatchNorm Layers. {default: True}

    Returns:
        Model: provided model as a bayesian
    """
    def replace_layers(model, init, prior, delta, skip_n_firsts=0, position=0):
        for name, layer in model.named_children():
            if layer.__class__ in DROPOUTS and auto_disable_dropout:
                    layer.eval()
            position = replace_layers(layer, init, prior, delta, skip_n_firsts, position + 1)
            if position >= skip_n_firsts:
                if layer.__class__ in TORCH2BAYE.keys():
                    params = init, prior, delta, freeze
                    bayesian = TORCH2BAYE[layer.__class__]
                    bayesian = bayesian.from_frequentist(layer, *params)
                    setattr(model, name, bayesian)
        return position

    new_model = deepcopy(model)
    replace_layers(new_model, initialization, prior, delta, skip_n_firsts)
    new_model = Model(model=new_model)
    
    return new_model