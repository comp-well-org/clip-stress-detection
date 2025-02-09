import torch
import torch.nn as nn
import torch.nn.functional as F
from .differentiable_augs import jitter, scaling, rotation, time_distortion, permutation
from .differentiable_augs import magnitude_warp, freq_depress
import numpy as np

class AugAttn(nn.Module):
    def __init__(self, channels, custom_init=True):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels=channels, out_channels=20, kernel_size=1, 
            stride=1, padding=0,
        )
        self.conv_2 = nn.Conv1d(
            in_channels=20, out_channels=20, kernel_size=1, 
            stride=1, padding=0,
        )
        self.conv_3 = nn.Conv1d(
            in_channels=20, out_channels=20, kernel_size=1, 
            stride=1, padding=0,
        )
        self.conv_4 = nn.Conv1d(
            in_channels=20, out_channels=channels, kernel_size=1, 
            stride=1, padding=0,
        )
        if custom_init:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    m.weight.data.normal_(0.0, 1.0)
                    m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.sigmoid(self.conv_4(x))
        return x

class GateNet(nn.Module):
    def __init__(self, num_channel, num_augmentations, layers=4) -> None:
        super().__init__()
        self.conv_blocks = []
        inchannel = num_channel
        outchannel = 32
        for _ in range(layers):
            self.conv_blocks.append(self._bulid_conv_block(inchannel, outchannel, 2, 1))
            inchannel = outchannel
            outchannel *= 2
        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(inchannel, num_augmentations)
        
    def _bulid_conv_block(self, in_channels, out_channels, kernel_size, stride, padding=0):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size, stride=stride, padding=padding,
            ),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AutoAUG(nn.Module):
    def __init__(self, num_channel, jitter_sigma=0.03, scaling_sigma=0.03):
        super().__init__()
        self.jitter_sigma = nn.Parameter(jitter_sigma * torch.ones(1), requires_grad=True)
        self.jitter = jitter

        self.scaling_sigma = nn.Parameter(scaling_sigma * torch.ones(1), requires_grad=True)
        self.scaling = scaling

        self.rotation_prob = nn.Parameter((1 - 0.1) * torch.ones(num_channel), requires_grad=True)
        self.rotation = rotation

        self.mixture_weights = nn.Parameter(torch.ones(5), requires_grad=True)
        self.nromal_mean = nn.Parameter(
            torch.Tensor([-0.5, -0.25, 0.0, 0.25, 0.5]), requires_grad=True,
        )
        self.nromal_sigma = nn.Parameter(
            torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7]), requires_grad=True,
        )
        self.timeDis = time_distortion

        self.permuation_seg = nn.Parameter(5 * torch.ones(1), requires_grad=True)
        self.permutation = permutation

        self.magW_sigma = nn.Parameter(0.1 * torch.ones(1), requires_grad=True)
        self.magW = magnitude_warp
        
        self.freq_mask_sigma = nn.Parameter(1 * torch.ones(1), requires_grad=True)
        self.freq_depress = freq_depress
        
        self.gateNet = GateNet(num_channel, 5)

        self.augAttn = AugAttn(channels=num_channel)
        self.params = [
            self.jitter_sigma, self.scaling_sigma, self.rotation_prob, self.mixture_weights,
            self.nromal_mean, self.nromal_sigma, self.freq_mask_sigma, 
            self.permuation_seg, self.magW_sigma,
        ]

        self.e = 1e-5
        self.steepness = 10.0  # Hyperparameter for relaxed thresholding
        self.gate_probs = []

    def normalization(self, x):
        x -= x.min(2, keepdim=True)[0]
        x /= (x.max(2, keepdim=True)[0] + 0.00000001)
        return x
    
    def monitor_gate_weights(self):
        output = np.array(self.gate_probs).mean(0)
        self.gate_probs = []
        return output
    
    def forward(self, x):
        gating_weights = self.gateNet(x)
        gate_probs = torch.sigmoid(
            self.steepness * (gating_weights - 0.5),
        ).unsqueeze(-1).unsqueeze(-1)
        self.gate_probs.append(gate_probs.mean(0).detach().cpu().numpy().flatten())
        
        x_jitter = self.jitter(x, 0.10 * torch.sigmoid(self.jitter_sigma), attention=None)
        x = gate_probs[:, 0] * x_jitter + (1 - gate_probs[:, 0]) * x  # Mixing
        x_scaling = self.scaling(x, 0.10 * torch.sigmoid(self.scaling_sigma)) 
        x = gate_probs[:, 1] * x_scaling + (1 - gate_probs[:, 1]) * x
        x_freq_dep = self.freq_depress(x, self.freq_mask_sigma).real
        x = gate_probs[:, 2] * x_freq_dep + (1 - gate_probs[:, 2]) * x 
        x_mag_warp = self.magW(x, 0.10 * torch.sigmoid(self.magW_sigma)) 
        x = gate_probs[:, 3] * x_mag_warp + (1 - gate_probs[:, 3]) * x
        x_time_dist = self.timeDis(
            x, self.mixture_weights, self.nromal_mean, F.relu(self.nromal_sigma) + self.e,
        )
        x = gate_probs[:, 4] * x_time_dist + (1 - gate_probs[:, 4]) * x
        
        x = self.normalization(x)
        return x