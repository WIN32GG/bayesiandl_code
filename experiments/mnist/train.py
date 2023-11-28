#!/bin/python3
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from bayeformers import to_bayesian
import bayeformers.nn as bnn
import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
import string, random, json
from uncertainty_metrics.numpy.general_calibration_error import ece
from scipy.special import softmax
import matplotlib.pyplot as plt
from torch import autograd

# torch.autograd.set_detect_anomaly(True)

class Dumper:
    
    def __init__(self, filename: str = None) -> None:
        if not filename:
            filename = f'unnamed.dump'
        self.original_filename : str = filename
        self.data = {}
        self.open()

    def reset(self):
        self.root_section    = Section(None)
        self.current_section = self.root_section

    def __call__(self, name: str = None, value = None):
        if name is None:
            if len(self.data) > 0:
                self.file_handle.write(json.dumps(self.data)+'\n')
                self.data = {}
        else:
            self.file_handle.write(f'#{name}\n')
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
        self.current_section = self.current_section.parent
        if self.current_section.is_root():
            self.dump()

    def open(self):
        postfix = "".join([random.choice(string.ascii_letters + string.digits) for n in range(5)]).upper()
        self.filename = f'{self.original_filename}.{postfix}'
        print(f'Dumping results to {self.filename}')
        self.file_handle = open("/dev/null", 'w+') 
            
    def dump(self):
        posfix = "".join([random.choice(string.ascii_letters + string.digits) for n in range(5)]).upper()
        self.filename = f'{self.original_filename}.{self.section_name}.{postfix}'
        print(f'Dumping results to {self.filename}')
        with open(self.filename, 'w+') as fh:
            fh.write(repr(self.root_section))
        print("Done dumping results")
        self.reset()

    def __setitem__(self, name: str, value):
        if type(value) == torch.Tensor:
            value = value.tolist()
        self.data[name] = value
        
# AVUC Utils

F32_ε  = 1.1920929e-07

entropy = lambda p: -1 * torch.sum(p * torch.log(p + F32_ε), axis=1) # BATCH_SIZE, C
Σ       = lambda e, ids: 0 if len(ids) == 0 else e[list(ids)].sum()
inter   = lambda a, b  : a.intersection(b)

# Constants
EPOCHS     = 3
SAMPLES    = 10
BATCH_SIZE = 128
LR         = 10e-3
N_CLASSES  = 10
W, H       = 28, 28
δ          = 1 # Conversion
α          = 1     # ELBO
β          = 3     # AVUC
γ          = 1     # NLL

log     = torch.log

def loss_avuc(p, y, uth=0.5, apply_softmax=True): # probs: for frequentist: direct, bayesian: mean
    """
        probs:  BATCH_SIZE, ELEMS
        labels: BATCH_SIZE
    """

    if apply_softmax:
        p     = F.softmax(p, dim=1)
    u     = entropy(p)
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


    return log( 1 + (( nau + nic ) / (nac + niu)) )
    

# Model Class Definition (Frequentist Implementation)
class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, n_classes: int) -> None:
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden,      hidden), nn.ReLU(),
            nn.Linear(hidden,   n_classes), nn.Softmax(dim=1),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp(input)

dumper     = Dumper("mnist_dump")
writer     = SummaryWriter("./logs/みこ")

# Dataset
transform = ToTensor()
dataset = MNIST(root="dataset", train=True, download=True, transform=transform)
dataset_test = MNIST(root="dataset", train=False, download=True, transform=transform)
loader = DataLoader(dataset,
    shuffle=True, batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=True,
)

loader_test = DataLoader(dataset_test,
    shuffle=True, batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=True,
)


def avuc_calibration(p):
    return entropy(F.softmax(p, dim=1)).mean()

def get_calibration(bmodel, loader):
    probabilities = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Calibration")
        for img, label in pbar:
            img, label = img.float().cuda(), label.long().cuda()

            # Setup Outputs
            prediction = torch.zeros(SAMPLES, img.size(0), 10).cuda()

            # Sample Loop (VI)
            for s in range(SAMPLES):
                prediction[s] = torch.softmax(bmodel(img.view(img.size(0), -1)), 1)
            
            prediction = prediction.mean(0).cpu().numpy()
            for p in prediction:
                probabilities.append(p.tolist())
    return avuc_calibration(torch.tensor(probabilities))

def get_ece(bmodel, loader):
    probabilities = []
    labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="ECE")
        for img, label in pbar:
            img, label = img.float().cuda(), label.long().cuda()

            # Setup Outputs
            prediction = torch.zeros(SAMPLES, img.size(0), 10).cuda()

            # Sample Loop (VI)
            for s in range(SAMPLES):
                prediction[s] = bmodel(img.view(img.size(0), -1))
            
            prediction = prediction.mean(0).cpu().numpy()

            for p in prediction:
                probabilities.append(p.tolist())
            for p in label:
                labels.append(p.tolist())
    e = ece(labels, probabilities)
    print(e)
    return e


# Model and Optimizer
model = MLP(W * H, 512, N_CLASSES).cuda() # Frequentist
optim = Adam(model.parameters(), lr=LR)   # Adam Optimizer

# Main Loop
with dumper("frequentist"):
    for epoch in tqdm(range(0), desc="Epoch"):
        tot_loss = 0.0
        tot_acc = 0.0

        # Batch Loop
        pbar = tqdm(loader, desc="Batch")
        for img, label in pbar:
            img, label = img.float().cuda(), label.long().cuda()

            # Reset Grads
            optim.zero_grad()

            # Loss Computation
            prediction = model(img.view(img.size(0), -1))
            loss = F.nll_loss(prediction, label, reduction="sum")
            acc = (torch.argmax(prediction, dim=1) == label).sum()

            with dumper():
                dumper['predictions'] = prediction
                dumper['label']       = label
            # Weights Update
            loss.backward()
            optim.step()

            # Display
            tot_loss += loss.item() / len(loader)
            tot_acc += acc.item() / len(dataset) * 100

            pbar.set_postfix(loss=tot_loss, acc=f"{tot_acc:.2f}%")

# Convertion to Bayesian
bmodel = to_bayesian(model, delta=δ, freeze=False)
bmodel = bmodel.cuda()

tot_loss = 0.0
tot_nll = 0.0
tot_log_prior = 0.0
tot_log_variational_posterior = 0.0
tot_acc = 0.0

ece_array = []

ece_array.append(get_ece(bmodel, loader_test))

# Batch Loop Bayesian Eval
pbar = tqdm(loader, desc="Bayesian Eval")
for img, label in pbar:
    break
    img, label = img.float().cuda(), label.long().cuda()

    # Setup Outputs
    prediction = torch.zeros(SAMPLES, img.size(0), 10).cuda()
    log_prior = torch.zeros(SAMPLES).cuda()
    log_variational_posterior = torch.zeros(SAMPLES).cuda()

    # Sample Loop (VI)
    for s in range(SAMPLES):
        prediction[s]   = bmodel(img.view(img.size(0), -1))
        log_prior[s]    = bmodel.log_prior()
        log_variational_posterior[s] = bmodel.log_variational_posterior()

    # Loss Computation
    log_prior = log_prior.mean()
    log_variational_posterior = log_variational_posterior.mean()
    nll = F.nll_loss(prediction.mean(0), label, reduction="sum")
    loss = (log_variational_posterior - log_prior) / len(loader) + nll
    acc = (torch.argmax(prediction.mean(0), dim=1) == label).sum()

    # Display
    nb = len(loader)
    tot_loss += loss.item() / nb
    tot_nll += nll.item() / nb
    tot_log_prior += log_prior.item() / nb
    tot_log_variational_posterior += log_variational_posterior.item() / nb
    tot_acc += acc.item() / len(dataset) * 100

    nb = len(loader)
    pbar.set_postfix(
        loss=tot_loss,
        nll=tot_nll,
        log_prior=tot_log_prior,
        log_variational_posterior=tot_log_variational_posterior,
        acc=f"{tot_acc:.2f}%"
    )
dumper()
# Main Loop

log_prior = torch.zeros(1).cuda()
log_variational_posterior = torch.zeros(1).cuda()    
optim = Adam(bmodel.parameters(), lr=LR)   # Adam Optimizer

with dumper('Bayesian'):
    step = 0
    for epoch in tqdm(range(2), desc="Epoch"):
        tot_loss = 0.0
        tot_nll = 0.0
        tot_log_prior = 0.0
        tot_log_variational_posterior = 0.0
        tot_acc = 0.0

        uth = get_calibration(bmodel, loader)
        print('uth=', uth)

        # Batch Loop Bayesian Train
        pbar = tqdm(loader, desc="Bayesian")
        for img, label in pbar:
            step += 1
            img, label = img.float().cuda(), label.long().cuda()

            # Reset grad
            optim.zero_grad()

            # Reset Logs
            log_prior *= 0
            log_variational_posterior *= 0 

            # Setup Outputs
            prediction = torch.zeros(SAMPLES, img.size(0), 10, requires_grad=True).cuda()

            # Sample Loop (VI)
            for s in range(SAMPLES):
                # with autograd.detect_anomaly():
                prediction[s] = bmodel(img.view(img.size(0), -1))
                log_prior += bmodel.log_prior()
                log_variational_posterior += bmodel.log_variational_posterior()

            # with dumper():
            #     dumper['predictions'] = prediction
            #     dumper['label']       = label

            # Loss Computation
            log_prior /= SAMPLES
            log_variational_posterior /= SAMPLES

            pred_mean = prediction.mean(0) + F32_ε
            nll  = F.nll_loss(torch.log(pred_mean), label, reduction="sum") # sad ε
            avuc = loss_avuc(pred_mean, label, uth, apply_softmax=True)
            elbo = (log_variational_posterior - log_prior) / len(loader)
            loss = α * elbo + γ * nll + β * avuc
            acc  = (torch.argmax(pred_mean, dim=1) == label).sum()

            # Weights Update
            loss.backward()
            optim.step()
            
            writer.add_scalar("log_prior", log_prior.item(), step)
            writer.add_scalar("log_variational_posterior", log_variational_posterior.item(), step)
            writer.add_scalar("nll", nll.item(), step)
            writer.add_scalar("avuc", avuc.item(), step)
            writer.add_scalar("total", loss.item(), step)

            # Display
            nb = len(loader)
            tot_loss += loss.item() / nb
            tot_nll += nll.item() / nb
            tot_log_prior += log_prior.item() / nb
            tot_log_variational_posterior += log_variational_posterior.item() / nb
            tot_acc += acc.item() / len(dataset) * 100

            nb = len(loader)
            pbar.set_postfix(
                loss=tot_loss,
                nll=tot_nll,
                log_prior=tot_log_prior,
                log_variational_posterior=tot_log_variational_posterior,
                acc=f"{tot_acc:.2f}%"
            )
        ece_array.append(get_ece(bmodel, loader_test))

torch.save(bmodel.state_dict(), "./mlp.pth")



