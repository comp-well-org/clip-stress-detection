import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def jitter(x, sigma, attention=None):
    m = torch.distributions.normal.Normal(loc=0, scale=sigma[0])
    noise = m.rsample(sample_shape=x.size())
    if attention is not None:
        noise = attention * noise
    noisy_x = x + noise
    return noisy_x

## Scaling
def scaling(x, sigma, attention=None):
    m = torch.distributions.normal.Normal(loc=1, scale=sigma[0])
    factor = m.rsample(sample_shape=x.size()[:2]).unsqueeze(-1)
    # if attention is not None:
    #     factor = torch.ones(factor.size()) + (factor - torch.ones(factor.size())) * attention
    output = x * factor
    return output