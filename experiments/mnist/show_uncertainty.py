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
import numpy as np
from torch import autograd
from scipy.ndimage import rotate

F32_ε  = 1.1920929e-07

entropy = lambda p: -1 * torch.sum(p * torch.log(p + F32_ε), axis=1) # BATCH_SIZE, C
Σ       = lambda e, ids: 0 if len(ids) == 0 else e[list(ids)].sum()
inter   = lambda a, b  : a.intersection(b)

# Constants
EPOCHS     = 3
SAMPLES    = 50
BATCH_SIZE = 1
LR         = 10e-3
N_CLASSES  = 10
W, H       = 28, 28
δ          = 1 # Conversion
α          = 1     # ELBO
β          = 3     # AVUC
γ          = 1     # NLL

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

model = MLP(W * H, 512, N_CLASSES)
model = to_bayesian(model, delta=0.4).cuda()
model.load_state_dict(torch.load("./mlp.pth"))
### Show uncertainty


entropy = lambda p: -1 * torch.sum(p * torch.log(p))

for img, label in loader_test:
    im = img[0][0]
    for r in range(0, 360, 15):
        img_rotate = rotate(im, r, reshape=False)
        prediction = torch.zeros(SAMPLES, 10).cuda()
        input = torch.unsqueeze(torch.unsqueeze(torch.tensor(img_rotate), dim=0), dim=0).cuda()
        input = input.view(input.size(0), -1).cuda()
        for i in range(SAMPLES):
            prediction[i] = model(input)
        mean_pred = prediction.mean(dim=0)
        uncertainty = entropy(mean_pred)
        pred = mean_pred.argmax()
        print(pred, uncertainty)
        plt.imshow(img_rotate, cmap='gray')
        plt.show()