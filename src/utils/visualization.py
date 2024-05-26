import torch
import matplotlib
import matplotlib.cm
from scipy.io import loadmat
import numpy as np
from pathlib import Path
seg_color_map = loadmat(Path(__file__).parent /'color150.mat')['colors']

def colorize_seg(seg_image):
    transposed = False
    batched = True
    if len(seg_image.shape) == 3:
        seg_image = seg_image[None]
        batched = False

    if seg_image.shape[-1] == 1:
        transposed = True
        seg_image = seg_image.permute(0, 3, 1, 2)


    # Bx3
    scmap = torch.from_numpy(seg_color_map).to(seg_image.device)
    N, _, H, W = seg_image.shape

    color_map = scmap[seg_image.view(-1)].view(N, H, W, 3).permute(0, 3, 1, 2) / 255.0

    if transposed:
        color_map = color_map.permute(0, 2, 3, 1)

    if not batched:
        color_map = color_map[0]

    return color_map


def colorize(grayscale_image, vmin=None, vmax=None, cmap='viridis', mask=None):
    '''
    Given Nx1xHxW binary image, return color mapped Nx3xHxW image
    '''
    transposed = False
    batched = True
    if len(grayscale_image.shape) == 3:
        grayscale_image = grayscale_image[None]
        batched = False

    if grayscale_image.shape[-1] == 1:
        transposed = True
        grayscale_image = grayscale_image.permute(0, 3, 1, 2)

    N, _, H, W = grayscale_image.shape
    min_v = grayscale_image.min()
    max_v = grayscale_image.max()
    dev = grayscale_image.device
    if not (vmin is None):
        grayscale_image = grayscale_image.clamp_min(vmin)
        min_v = vmin
    if not (vmax is None):
        grayscale_image = grayscale_image.clamp_max(vmax)
        max_v = vmax
    bitmap = ((grayscale_image - min_v) / (max_v - min_v) * 255).byte()
    c = torch.Tensor(matplotlib.cm.get_cmap(cmap).colors).to(dev)
    color_map = c.index_select(0, bitmap.long().view(-1)).view(N, H, W, 3).permute(0, 3, 1, 2)
    if not (mask is None):
        color_map[mask.repeat(1, 3, 1, 1)] = 0
    if transposed:
        color_map = color_map.permute(0, 2, 3, 1)

    if not batched:
        color_map = color_map[0]

    return color_map


def translate(normal):
    return (normal + 1) / 2

