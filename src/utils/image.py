import numpy as np
import torch
from tqdm import tqdm, trange
from .visualization import colorize, colorize_seg

def combine_renderables(gts, infs):
    H, W = gts['rgb'].shape[:2]
    device = gts['rgb'].device
    # render based on gts / infs
    shared_keys = [key for key in ['rgb', 'd', 'n'] if (key in gts) and (key in infs)]
    non_shared_inf_keys = [key for key in infs.keys() if key not in shared_keys]
    non_shared_gt_keys = [key for key in gts.keys() if key not in shared_keys]
    max_num_renderable = len(shared_keys) + max(len(non_shared_gt_keys), len(non_shared_inf_keys))
    renderable = torch.zeros(H*2, W*max_num_renderable, 3).to(device)

    for i, key in enumerate(shared_keys):
        renderable[:H, i*W:(i+1)*W] = gts[key]
        renderable[H:, i*W:(i+1)*W] = infs[key]
    for j, key in enumerate(non_shared_gt_keys):
        i = len(shared_keys) + j
        renderable[:H, i*W:(i+1)*W] = gts[key]
    for j, key in enumerate(non_shared_inf_keys):
        i = len(shared_keys) + j
        renderable[H:, i*W:(i+1)*W] = infs[key]

    return renderable.permute(2, 0, 1)

def render_gt_image(batch, H, W):
    gts = {
        'rgb': batch['rgb'].view(H, W, 3),
    }

    has_gt_normal= 'n' in batch
    has_gt_mask = 'm' in batch
    has_gt_depth = 'd' in batch
    has_gt_seg = 'seg' in batch

    if has_gt_normal:
        gts['n'] = batch['n'].view(H, W, 3) * 0.5 + 0.5
    if has_gt_depth:
        gts['d'] = colorize(batch['d'].view(H, W, 1))
    if has_gt_mask:
        gts['m'] = batch['m'].view(H, W, 1).repeat(1, 1, 3)
    if has_gt_seg:
        gts['seg'] = colorize_seg(batch['seg'].view(H, W, -1))
    return gts

def render_image(batch, model, grid, shape, chunk_size=1024, fp_16=True):
    infs = render_image_raw(batch, model, grid, shape, chunk_size, fp_16)
    has_normal= 'n' in infs
    has_depth = 'd' in infs
    has_density_normal = 'gn' in infs

    if has_depth:
        infs['d'] = colorize(infs['d'])

    if has_normal:
        infs['n'] = infs['n'] * 0.5 + 0.5

    if has_density_normal:
        infs['gn'] = infs['gn'] * 0.5 + 0.5
    return infs

def render_image_raw(batch, model, grid, shape, chunk_size=1024, fp_16=True):
    num_rays = len(batch['ray_o'])

    inf_rgbs = []
    inf_ds = []
    inf_ns = []
    inf_sems = []
    inf_gns = []
    has_normal = False
    has_density_normal = False
    has_sem = False
    pose = batch['pose']

    R = pose[:3, :3]

    for bs in trange(0, num_rays, chunk_size, dynamic_ncols=True, leave=False):
        # chunk batches
        chunk_batch = {}
        for k, v in batch.items():
            if k not in ['pose', 'iid', 'inds']:
                chunk_batch[k] = v[bs:bs+chunk_size]
            else:
                chunk_batch[k] = v

        ray_o = chunk_batch['ray_o']
        ray_d = chunk_batch['ray_d']
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=fp_16):
                ray_indices, t_start, t_ends = grid.sample(ray_o, ray_d)
                res = model.render(ray_o, ray_d, ray_indices, t_start, t_ends)

            inf_rgb = res['rgb']
            inf_d = res['d']
            if 'n' in res:
                inf_n = res['n']
                inf_ns.append(inf_n)
                has_normal = True
            if 'gn' in res:
                inf_gn = res['gn']
                inf_gns.append(inf_gn)
                has_density_normal = True
            if 'seg' in res:
                inf_sl = res['seg']
                inf_s = torch.argmax(inf_sl, dim=-1)
                inf_sems.append(inf_s)
                has_sem = True

            inf_rgbs.append(inf_rgb)
            inf_ds.append(inf_d)
    inf_rgbs = torch.cat(inf_rgbs)
    inf_ds = torch.cat(inf_ds)
    ret = {
        'rgb': inf_rgbs.view(*shape, 3),
        'd': inf_ds.view(*shape, 1)
    }

    if has_normal:
        ret['n'] = (torch.cat(inf_ns).view(*shape, 3) @ R)

    
    if has_density_normal:
        ret['gn'] = (torch.cat(inf_gns).view(*shape, 3) @ R)

    if has_sem:
        inf_sems = torch.cat(inf_sems).view(*shape, -1)
        ret['seg'] = colorize_seg(inf_sems)
        
    return ret
