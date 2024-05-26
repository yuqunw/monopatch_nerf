import os
import sys
sys.path.append('.')
sys.path.append('..')

from pathlib import Path
import json
from scripts.fusion import fuse_reconstruction
from PIL import Image
from lpips import LPIPS
from pytorch_msssim import ssim
import torchvision.transforms.functional as F
import torch
import numpy as np
from tqdm import tqdm
import math

eth3d_evaluation_bin = Path('/home/yuqunwu2/large_scale_nerf/multi-view-evaluation/build/ETH3DMultiViewEvaluation')

scenes = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']

def evaluate_3d(pc_file, gt_path):
    gt_file = gt_path / "dslr_scan_eval" / "scan_alignment.mlp"
    exe_str = f'{str(eth3d_evaluation_bin)} --reconstruction_ply_path {pc_file} --ground_truth_mlp_path {gt_file} --tolerances 0.02,0.05'
    output = os.popen(exe_str).read()
    lines = output.split('\n')
    tolerances = [0.02, 0.05]
    com_index = [i for i, line in enumerate(lines) if line.find('Completeness') == 0][0]
    com_line = lines[com_index]
    acc_line = lines[com_index+1]
    f1_line = lines[com_index+2]
    com_words = com_line.split()
    acc_words = acc_line.split()
    f1_words = f1_line.split()
    ress = {}
    for i, tol in enumerate(tolerances):
        res ={}
        res[f'completeness'] = float(com_words[i + 1])
        res[f'accuracy'] = float(acc_words[i + 1])
        res[f'f1'] = float(f1_words[i + 1])
        ress[f'tol_{tol}'] = res

    return ress

def measure_psnr(ref_image, src_image):
    mse = ((ref_image - src_image) ** 2).mean()
    return -10.0 * math.log10(mse.item())

lpips_fn = LPIPS(net='alex').to('cuda').eval()

def measure_lpips(ref_image, src_image):
    with torch.no_grad():
        return lpips_fn(ref_image[None].cuda(), src_image[None].cuda(), normalize=True).cpu().item()

def measure_ssim(ref_image, src_image):
    return ssim(ref_image[None], src_image[None], data_range=1.0).item()

def evaluate_images(input_path, result_path):
    # -- evaluate test
    with open(input_path / 'transforms_test.json') as f:
        trans = json.load(f)
    frames = trans['frames']
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    print(f'Evaluating Test Images', flush=True)
    for i, frame in enumerate(frames):
        print(f'Evaluated {i+1}/{len(frames)}', flush=True)
        gt_file_path= input_path / frame['file_path']
        inf_file_path = result_path / frame['file_path']
        gt_image = Image.open(gt_file_path)
        inf_image = Image.open(inf_file_path)

        gt_rgb = F.to_tensor(gt_image)
        inf_rgb = F.to_tensor(inf_image)

        # compute PSNR, SSIM, LPIPS
        psnr_val = measure_psnr(inf_rgb, gt_rgb)
        ssim_val = measure_ssim(inf_rgb, gt_rgb)
        lpips_val = measure_lpips(inf_rgb, gt_rgb)

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        lpips_vals.append(lpips_val)

    # compute average and return
    test_psnr = {
        'test/psnr': np.array(psnr_vals).mean(),
        'test/ssim': np.array(ssim_vals).mean(),
        'test/lpips': np.array(lpips_vals).mean(),
    }

    # -- evaluate train
    with open(input_path / 'transforms_train.json') as f:
        trans = json.load(f)
    frames = trans['frames']
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    print(f'Evaluating Non-test Images', flush=True)
    for i, frame in enumerate(frames):
        print(f'Evaluated {i+1}/{len(frames)}', flush=True)
        gt_file_path= input_path / frame['file_path']
        inf_file_path = result_path / frame['file_path']
        gt_image = Image.open(gt_file_path)
        inf_image = Image.open(inf_file_path)

        gt_rgb = F.to_tensor(gt_image)
        inf_rgb = F.to_tensor(inf_image)

        # compute PSNR, SSIM, LPIPS
        psnr_val = measure_psnr(inf_rgb, gt_rgb)
        ssim_val = measure_ssim(inf_rgb, gt_rgb)
        lpips_val = measure_lpips(inf_rgb, gt_rgb)

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        lpips_vals.append(lpips_val)

    # compute average and return
    train_psnr = {
        'train/psnr': np.array(psnr_vals).mean(),
        'train/ssim': np.array(ssim_vals).mean(),
        'train/lpips': np.array(lpips_vals).mean(),
    }
    return {**train_psnr, **test_psnr}

def main(args):
    # first perform fusion
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    gt_path = Path(args.gt_path)
    pc_file = output_path / 'results' / 'fused.ply'

    # evaluate images for test samples
    evals_images = evaluate_images(input_path, output_path)

    # evaluate 3d 
    evals_3d = evaluate_3d(pc_file, gt_path)

    # write evaluation results to file
    evals = {**evals_images, **evals_3d}

    with open(output_path / 'results' / 'results.json', 'w') as f:
        json.dump(evals, f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--threshold', default=2.0, type=float)
    parser.add_argument('--min_views', default=2, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    main(args)
