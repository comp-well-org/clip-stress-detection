import torch
import torch.nn.functional as F
from .distribution.stable_nromal import StableNormal
from .distribution.mixture_same_family import ReparametrizedMixtureSameFamily

class RoundBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class FloatBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return float(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def jitter(x, sigma, attention=None):
    m = torch.distributions.normal.Normal(loc=0, scale=sigma[0])
    noise = m.rsample(sample_shape=x.size())
    if attention is not None:
        noise = attention * noise
    noisy_x = x + noise
    return noisy_x

def scaling(x, sigma, attention=None):
    m = torch.distributions.normal.Normal(loc=1, scale=sigma[0])
    factor = m.rsample(sample_shape=x.size()[:2]).unsqueeze(-1)
    output = x * factor
    return output

def rotation(x, probs, temperature=0.01):
    m = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(temperature, probs=probs)
    flip = m.rsample()
    flip = F.hardtanh(flip)
    output =  flip.view(x.size()[0], x.size()[1], 1) * x
    return output

def time_distortion(x, mixture_weights, nromal_mean, nromal_sigma):
    mixture_cate = torch.distributions.Categorical(probs=F.softmax(mixture_weights))
    normal_dists = StableNormal(loc=nromal_mean, scale=nromal_sigma)
    mixture_normal = ReparametrizedMixtureSameFamily(
        mixture_distribution=mixture_cate,
        component_distribution=normal_dists,
    )
    mixture_norm_samples = mixture_normal.rsample(sample_shape=(x.size(0), x.size(2)))
    mixture_norm_samples = torch.tanh(mixture_norm_samples)
    mixture_norm_samples, _ = torch.sort(mixture_norm_samples)

    x_hw = x.unsqueeze(2)
    grid_hw = mixture_norm_samples.unsqueeze(1)
    grid_hw = torch.cat([grid_hw, 0 * torch.ones_like(grid_hw)], dim=1).permute(0, 2, 1).unsqueeze(1)
    x_distorted = F.grid_sample(x_hw, grid_hw, align_corners=False).squeeze(-2)
    return x_distorted

class PermuteView(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_seg):
        splits = torch.split(x, torch.div(x.size(-1), int(num_seg), rounding_mode='trunc'), dim=-1)
        if x.size(-1) % int(num_seg) == 0:
            permuted_order = torch.randperm(int(num_seg))
        else:
            permuted_order = torch.randperm(int(num_seg) + 1)
        ctx.save_for_backward(num_seg, permuted_order)
        splits_permuted = []
        for idx in permuted_order.detach().numpy():
            splits_permuted.append(splits[idx])
        output = torch.cat(splits_permuted, dim=-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        num_seg, permuted_order, = ctx.saved_tensors
        splits = torch.split(
            grad_output, 
            torch.div(grad_output.size(-1), int(num_seg), rounding_mode='trunc'), 
            dim=-1,
        )
        splits_permuted = []
        for idx in permuted_order:
            splits_permuted.append(splits[idx])
        grad_output = torch.cat(splits_permuted, dim=-1)
        return grad_output, grad_output.mean().unsqueeze(0)

def permutation(x, max_segments):
    m_uniform = torch.distributions.uniform.Uniform(1, max_segments)
    num_segs_soft = m_uniform.rsample(x.size()[:1])
    num_segs = torch.round(num_segs_soft) - num_segs_soft.detach() + num_segs_soft
    permute_x = []
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            permute_view = PermuteView.apply(pat.unsqueeze(0), num_segs[i])
        else:
            permute_view = pat.unsqueeze(0)
        permute_x.append(permute_view)
    return torch.cat(permute_x, dim=0)

def magnitude_warp(x, sigma, knot=4):
    m = torch.distributions.normal.Normal(loc=1, scale=sigma[0])
    yy = m.rsample(sample_shape=(x.size(0), x.size(1), knot))
    wave = F.interpolate(yy, size=x.size(2), mode='linear')
    return wave * x

def freq_depress(x, rate, temperature=10.0, scale_factor=1.0, dim=1):
    xy_f = torch.fft.fft(x, dim=dim)
    x_device = x.get_device()
    
    m_value = (torch.FloatTensor(xy_f.shape).uniform_().to(x_device) - rate) * temperature
    m = torch.sigmoid(m_value)
    
    amp = torch.abs(xy_f)

    normalized_amp = amp / amp.max()
    tanh_input = scale_factor * normalized_amp
    protection_mask = 0.5 * (torch.tanh(tanh_input) + 1)
    
    combined_mask = (1 - protection_mask) * m

    freal = xy_f.real * (1 - combined_mask)
    fimag = xy_f.imag * (1 - combined_mask)
    xy_f = torch.complex(freal, fimag)
    
    xy = torch.fft.ifft(xy_f, dim=dim)
    return xy
