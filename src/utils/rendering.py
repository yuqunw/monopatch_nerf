import torch
from nerfacc import render_weight_from_density, accumulate_along_rays
from torch_efficient_distloss import flatten_eff_distloss
import torch.nn.functional as NF

# code from nerfacc.

def render(
    # ray marching results
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: torch.Tensor,
    n_rays: int,
    # radiance field
    query_fn,
    seg_fn,
    # rendering options
    render_bkgd=None,
    training=False
):
    # Query sigma/alpha and color with gradients
    qres = query_fn(t_starts, t_ends, ray_indices)

    sigmas = qres['s']
    d = (t_starts + t_ends) / 2.0


    weights, transmittance, opacities = render_weight_from_density( t_starts, t_ends, sigmas[..., 0], ray_indices=ray_indices, n_rays=n_rays)

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    res = {}
    res['rgb'] = accumulate_along_rays(weights, values=qres['rgb'], ray_indices=ray_indices, n_rays=n_rays).clamp(0, 1)
    inf_n = accumulate_along_rays( weights, ray_indices=ray_indices, values=qres['n'], n_rays=n_rays)

    inf_gn = accumulate_along_rays( weights, ray_indices=ray_indices, values=qres['gn'], n_rays=n_rays)
    res['n'] = NF.normalize(inf_n, p=2, dim=-1)
    res['gn'] = NF.normalize(inf_gn, p=2, dim=-1)

    # Lseg feats
    if seg_fn is not None:
        seg_feats = accumulate_along_rays( weights, ray_indices=ray_indices, values=qres['seg_f'], n_rays=n_rays)
        res['seg_f'] = seg_feats
        res['seg'] = seg_fn(seg_feats)
    
    res['a'] = accumulate_along_rays( weights, ray_indices=ray_indices, values=None, n_rays=n_rays)
    weight_sum = res['a'][ray_indices]
    norm_weights = weights / weight_sum[..., 0].clamp_min(1e-6)
    inf_d = accumulate_along_rays( norm_weights, ray_indices=ray_indices, values=d[..., None], n_rays=n_rays)
    res['d'] = inf_d

    # Background composition.
    if render_bkgd is not None:
        res['rgb'] = res['rgb'] + render_bkgd * (1.0 - res['a'])

    # if training:
    #    res['dist_loss'] = flatten_eff_distloss(norm_weights, (t_ends + t_starts) / 2.0, (t_ends - t_starts), ray_indices)

    return res
