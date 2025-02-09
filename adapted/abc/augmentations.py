import torch

def jitter(x, sigma, attention=None):
    m = torch.distributions.normal.Normal(loc=0, scale=sigma)
    noise = m.rsample(sample_shape=x.size())
    if attention is not None:
        noise = attention * noise
    noise = noise.to(x.device)
    noisy_x = x + noise
    return noisy_x

def scaling(x, sigma, attention=None):
    m = torch.distributions.normal.Normal(loc=1, scale=sigma)
    factor = m.rsample(sample_shape=x.size()[:2]).unsqueeze(-1)
    factor = factor.to(x.device)
    output = x * factor
    return output
