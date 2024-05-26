import torch.nn as nn
import torch
import math
from .qff_cuda import qff

class QFF(nn.Module):
    def __init__(self, input_width, min_log2_freq, max_log2_freq, num_freqs=16, quant_size=64, num_feats=1, std=0.0001, use_identity: bool = True):
        super().__init__()
        self.input_width =input_width 
        self.num_freqs = num_freqs

        self.num_freqs = num_freqs
        self.use_identity = use_identity
        self.num_feats = num_feats

        # freqs = torch.tensor([2**i for i in linspace(min_log2_freq, max_log2_freqs, num_freqs)])
        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)

        cv = torch.randn((num_freqs * 2, num_feats, quant_size, quant_size, quant_size)) * std
        self.cv = nn.Parameter(cv, True)
        self.quant_size = quant_size

    
    @property
    def latent_dim(self) -> int:
        return self.output_width

    @property
    def output_width(self) -> int:
        return (int(self.use_identity) * self.input_width) + self.num_freqs * 2 * self.num_feats

    def forward(self, points):
        """
        must return features given points (and optional dirs)
        """
        if any([s == 0 for s in points.shape]):
            return torch.zeros((0, self.latent_dim)).to(points)
        return qff.apply(points, self.freqs.to(points), self.cv)
