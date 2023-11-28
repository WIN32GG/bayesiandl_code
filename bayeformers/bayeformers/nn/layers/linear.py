# -*- coding: utf-8 -*-
"""bayeformers.nn.layers.linear

The files provides a Bayesian equivalent for the Linear module present in the
PyTorch library as torch.nn.Linear.
"""
from bayeformers.nn.parameters.base import NoneParameter
from bayeformers.nn.parameters.base import Parameter
from bayeformers.nn.parameters.gaussian import Gaussian
from bayeformers.nn.parameters.gaussian import DEFAULT_SCALED_GAUSSIAN_MIXTURE
from bayeformers.nn.parameters.gaussian import ScaledGaussianMixture
from bayeformers.nn.parameters.initializations import DEFAULT_UNIFORM
from bayeformers.nn.parameters.initializations import Initialization
from torch import Size
from torch import Tensor
from torch.nn import Module
from typing import Optional
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT_RHO     = np.log(np.e - 1)
F32_EPSILON  = 1.1920929e-07

class Linear(Module):
    """Linear

    Bayesian Linear module using Gaussian parameters and a Scaled Gaussian
    Mixture as default prior for the weights.

    Attributes:
        in_features (int): input number of features
        out_features (int): output number of features
        initialization (Optional[Initialization]): initialization callback
            for the gaussian parameters
        weight (Gaussian): gaussian weight of the linear layer
        weight_prior (Parameter): prior of the linear layer weight
        bias (Gaussian): gaussian bias of the linear layer
        bias_prior (Parameter): prior of the linear layer bias
        log_prior (torch.Tensor): log prior of the weight
        log_variational_posterior (torch.Tensor): log variational posterior of
            the weight
    """

    def __init__(
        self, in_features: int, out_features: int, bias: Optional[bool] = True,
        initialization: Optional[Initialization] = DEFAULT_UNIFORM,
        prior: Optional[Parameter] = DEFAULT_SCALED_GAUSSIAN_MIXTURE
    ) -> None:
        """Initialization

        Arguments:
            in_features (int): input number of features
            out_features (int): output number of features

        Keyword Arguments:
            bias (bool): presence of bias in the layer {default: True}
            initialization (Optional[Initialization]): initialization callback
                for the gaussian parameters {default: DEFAULT_UNIFORM}
            prior (Optional[Parameter]): prior of the weight
                {default: DEFAULT_SCALED_GAUSSIAN_MIXTURE}
        """
        super(Linear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.initialization = initialization

        size = Size((self.out_features, self.in_features))
        self.weight = Gaussian(size, self.initialization)
        self.weight_prior = ScaledGaussianMixture(0.5, np.exp(-0), np.exp(-6))
        
        if bias:
            size = Size((self.out_features, ))
            self.bias = Gaussian(size, self.initialization)
            self.bias_prior = ScaledGaussianMixture(0.5, np.exp(-0), np.exp(-6))
        else:
            self.bias = NoneParameter()
            self.bias_prior = NoneParameter()

        # self.log_prior = torch.zeros(1, requires_grad=True).cuda()
        # self.log_variational_posterior = torch.zeros(1, requires_grad=True).cuda()
        self.register_buffer("log_prior",                 torch.tensor(0., requires_grad = True), persistent=False)
        self.register_buffer("log_variational_posterior", torch.tensor(0., requires_grad = True), persistent=False)

    def forward(self, input: Tensor) -> Tensor:
        """Forward

        Feed forward pass of the linear layer.
        W ~ N(mu, rho)
        b ~ N(mu, rho)
        z = W x + b

        Args:
            input (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        weight, bias = self.weight.sample(), self.bias.sample()

        # REMOVE Debugging this mess        
        # print(self.weight.mu)
        # print(self.weight.rho)

        # print(weight)
        # print(bias)

        # print(self.weight_prior.log_prob(weight))
        # exit(1)

        # NB: Disregard previous values, and keep auto grad by *0

        # self.log_prior = self.log_prior.to(input.device)
        # self.log_variational_posterior = self.log_variational_posterior.to(input.device)

        # lp = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        # self.log_prior.copy_(lp)
        # lvp = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        # self.log_variational_posterior.copy_(lvp)

        self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)

        return F.linear(input, weight, bias=bias) 

    @classmethod
    def from_frequentist(
        cls, linear: Module,
        initialization: Optional[Initialization] = DEFAULT_UNIFORM,
        prior: Optional[Parameter] = DEFAULT_SCALED_GAUSSIAN_MIXTURE,
        delta: float = None, freeze: bool = False
    ) -> "Linear":
        """From Frequentist

        Build a bayesian linear layer out of a frequentist linear layer with
        the same parameters.

        Arguments:
            linear (Module): frequentist linear module

        Keyword Arguments:
            initialization (Optional[Initialization]): initialization callback
                for the gaussian parameters {default: DEFAULT_UNIFORM}
            prior (Optional[Parameter]): prior of the weight
                {default: DEFAULT_SCALED_GAUSSIAN_MIXTURE}
            delta (float): is the model pretrained? If it not None, the model
                weights will be used to initialize the bayesian weights and
                delta will be used for MOPED posterior init {default: None}
                pretrained loading following:
                    "Specifying Weight Priors in Bayesian Deep Neural Networks
                    with Empirical Bayes" from Krishnan et al.
                reference: https://arxiv.org/pdf/1906.05323.pdf
            freeze (bool): freeze weight's mu if delta is not None
                {default: False}
        """
        bias = linear.bias is not None
        baye = cls(linear.in_features, linear.out_features, bias, prior=prior)

        if delta is not None:
            baye.weight.mu.data.copy_(linear.weight.data, True)
            baye.weight.rho.data.copy_(torch.log(
                torch.exp(delta * torch.abs(linear.weight.data)) - 1.0 
            ), True).to(baye.weight.mu.device)
#             print(baye.weight.rho.data)
#             m = (baye.weight.rho.data == float('-inf')).float()
#             torch.set_printoptions(profile="full")
#             nz = (s*m).nonzero()
            
#             print(s[nz])
#             exit(1)
            
#             baye.weight.rho.data[baye.weight.rho.data == float("-inf")].float().sum()
#             print((baye.weight.rho.data == float("-inf")).float().sum())
            baye.weight.mu.requires_grad = not freeze
            baye.weight.rho.requires_grad = True 

            prior = Gaussian(baye.weight.mu.size())
            prior.mu.data.copy_(linear.weight.data)
            prior.rho.data = torch.ones_like(linear.weight) * UNIT_RHO
            # prior.mu.requires_grad = False  # Priors don't update
            # prior.rho.requires_grad = False # Priors don't update
            baye.weight_prior = prior
            
            if linear.bias is not None:
                baye.bias.mu.data.copy_(linear.bias.data, True)
                baye.bias.rho.data.copy_(torch.log(
                   torch.exp(delta * torch.abs(linear.bias.data)) - 1.0
                ), True).to(baye.bias.mu.device)
#                 print((baye.bias.rho.data == float("-inf")).float().sum())
#                 baye.bias.rho.data[baye.weight.rho.data == float("-inf")].float().sum()
                baye.bias.mu.requires_grad = not freeze
                baye.bias.rho.requires_grad = True
                
                prior = Gaussian(baye.bias.mu.size())
                prior.mu.data.copy_(linear.bias.data)
                prior.rho.data.copy_(torch.ones_like(linear.bias) * UNIT_RHO)
                # prior.mu.requires_grad = False  # Priors don't update
                # prior.rho.requires_grad = False # Priors don't update
                baye.bias_prior = prior

        return baye