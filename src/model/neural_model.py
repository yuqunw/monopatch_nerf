import torch
import torch.nn as nn
import torch.nn.functional as NF

from src.utils.arg_dict_class import ArgDictClass

import numpy as np
from .qff import QFF
from .encodings import trunc_exp
from src.utils.rendering import render


class NeuralModel(ArgDictClass, nn.Module):
    def __init__(self,
                # for encoding
                num_feats=4,
                min_log2_freq=0,
                max_log2_freq=5,
                num_freqs=4,
                num_quants=64,
                use_normal=True,
                use_density_normal=True,
                use_seg=True
        ):
        super().__init__()
        self.encoder = QFF(3, min_log2_freq, max_log2_freq, num_freqs, num_quants, num_feats, 0.0001, False)
        self.geom_mlp = nn.Sequential(
            nn.Linear(self.encoder.output_width, 16, bias=None),
            nn.ReLU(),
            nn.Linear(16, 16, bias=None)
        )
        self.density_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.color_mlp = nn.Sequential(
            nn.Linear(15 + 3, 16, bias=None),
            nn.ReLU(),
            nn.Linear(16, 3, bias=None)
        )
        if use_normal:
            self.normal_mlp = nn.Sequential(
                nn.Linear(15, 16, bias=None),
                nn.ReLU(),
                nn.Linear(16, 3, bias=None)
            )
        if use_seg:
            self.seg_mlp = nn.Sequential(
                nn.Linear(15, 16, bias=None),
                nn.ReLU(),
                nn.Linear(16, 16, bias=None)
            )
            self.seg_dec = nn.Sequential(
                nn.Linear(16, 150, bias=None)
            )   
        self.use_seg = use_seg
        self.use_normal = use_normal
        self.use_density_normal = use_density_normal

        
    @property
    def device(self):
        return self.transform.device

    def render(self, ray_o, ray_d, ray_i, t_starts, t_ends):
        B = len(ray_o)
        if len(t_starts) == 0:
            res = {}
            res['rgb'] = torch.zeros_like(ray_o)
            res['n'] = torch.zeros_like(ray_o)
            res['gn'] = torch.zeros_like(ray_o)
            res['a'] = torch.zeros_like(ray_o)[..., :1]
            res['d'] = torch.zeros_like(ray_o)[..., :1]
            return res

        def query_fn(t_starts, t_ends, ray_i):
            t_origins = ray_o[ray_i]
            t_dirs = ray_d[ray_i]
            points = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            return self.query(points, t_dirs)
        
        if self.use_seg:
            def seg_fn(seg_feats):
                return self.seg_dec(seg_feats)
        else:
            seg_fn = None
        return render( t_starts, t_ends, ray_i, n_rays=B, query_fn=query_fn, seg_fn=seg_fn, render_bkgd=None, training=self.training)
    
    def query(self, points, dirs):
        '''
        points: Bx3
        dirs: Bx3

        returns:
            colors: Bx3
            sdf: Bx1
            dsdf_dx: Bx3
        '''
        if self.use_density_normal:
            return self.query_with_grad(points, dirs)
        # 
        feats = self.encoder(points)
        s = self.geom_mlp(feats)
        density = trunc_exp(s[:, :1] + self.density_bias)
        rgbs = self.color_mlp(torch.cat((s[:, 1:], dirs), 1)).sigmoid()
        res = {
            'rgb': rgbs,
            's': density,
        }
        if self.use_normal:
            res['n'] = NF.normalize(self.normal_mlp(s[:, 1:]), p=2, dim=-1)
        if self.use_seg:
            res['seg_f'] = self.seg_mlp(s[:, 1:])
        return res


    def query_with_grad(self, points, dirs):
        '''
        points: Bx3
        dirs: Bx3

        returns:
            colors: Bx3
            sdf: Bx1
            dsdf_dx: Bx3
        '''
        point_requires_grad = points.requires_grad
        points.requires_grad_(True)
        context_requires_grad = torch.is_grad_enabled()

        with torch.enable_grad():
            feats = self.encoder(points)
            s = self.geom_mlp(feats)
            density = trunc_exp(s[:, :1] + self.density_bias)

            d_output = torch.ones_like(density, requires_grad=False, device=density.device)
            grad_normals = torch.autograd.grad(
                outputs=density,
                inputs=points,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            grad_normals = NF.normalize(-grad_normals, p=2, dim=-1, eps=1e-8)

        if not point_requires_grad:
            points = points.detach()

        if not context_requires_grad:
            density = density.detach()
            grad_normals = grad_normals.detach()
            feats = feats.detach()
            s = s.detach()

        rgbs = self.color_mlp(torch.cat((s[:, 1:], dirs), 1)).sigmoid()
        res = {
            'rgb': rgbs,
            's': density,
            'gn': grad_normals,
        }
        if self.use_normal:
            res['n'] = NF.normalize(self.normal_mlp(s[:, 1:]), p=2, dim=-1)

        if self.use_seg:
            res['seg_f'] = self.seg_mlp(s[:, 1:])
        
        return res

    def query_density(self, points):
        '''
        points: Bx3
        dirs: Bx3

        returns:
            colors: Bx3
            sdf: Bx1
            dsdf_dx: Bx3
        '''
        with torch.no_grad():
            feats = self.encoder(points)
            density = self.geom_mlp(feats)[:, :1] + self.density_bias
            return trunc_exp(density)
