import sys
sys.path.append('.')
sys.path.append('..')

import torch
import numpy as np
from src.utils.checkpoints import Checkpointer
from src.data.eval_datamodule import DataModule
import math
from pathlib import Path
from tqdm import tqdm
from src.utils.image import *
import cv2
import shutil
from PIL import Image
import json
import torch.nn.functional as NF

def main(args):
    datamodule = DataModule(args.output_path, device=args.device, full=args.full)

    model_ckpt = torch.load(args.model_checkpoint_file, args.device)
    model = model_ckpt['class'](**model_ckpt['arg_dict'])
    model.load_state_dict(model_ckpt['state_dict'])
    model = model.eval().to(args.device)

    grid_ckpt = torch.load(args.grid_checkpoint_file, args.device)
    grid = grid_ckpt['class'](**grid_ckpt['arg_dict'])
    grid.poses = grid_ckpt['poses']
    grid.load_state_dict(grid_ckpt['state_dict'])
    grid = grid.eval().to(args.device)

    # write out
    '''
    output_path/
        images/
            000000.rgb.png
        depths/
            000000.depth.tiff
        transforms.json
    '''
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'depths').mkdir(parents=True, exist_ok=True)
    (output_path / 'normals').mkdir(parents=True, exist_ok=True)

    # shutil.copy(Path(args.data_path) / 'transforms.json', output_path / 'transforms.json')


    datamodule.setup_test()
    dataset = datamodule.dataset

    H, W = datamodule.height, datamodule.width
    batch_size = 1024
    fp_16 = False
    
    # iterate and write to depth
    print('Evaluating!')
    for i, batch in enumerate(dataset):
        infs = render_image_raw(batch, model, grid, (H, W), batch_size, fp_16)
        rgb = infs['rgb']
        depth = infs['d']
        scale = batch['ray_s'].view(*depth.shape)
        pose = batch['pose']
        scaled_depth = depth / scale
        # normals = NF.normalize(infs['n'].view(*depth.shape, 3), p=2, dim=-1)
        # normals = (normals + 1) / 2
        # normals = normals.clamp(0, 1)

        inf_dn = NF.normalize(infs['gn'].view(*depth.shape, 3), p=2, dim=-1)
        inf_dn[..., 1] = -inf_dn[..., 1] # For visualization
        inf_dn[..., 2] = -inf_dn[..., 2] # For visualization
        inf_dn = (inf_dn + 1) / 2
        np_image = (rgb.view(H, W, 3).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(np_image).save(output_path / 'images' / f'{i:06d}.rgb.png')
        cv2.imwrite(str(output_path / 'depths' / f'{i:06d}.depth.tiff'), scaled_depth.cpu().numpy())
        Image.fromarray((inf_dn.view(H, W, 3).cpu().numpy() * 255).astype(np.uint8)).save(output_path / 'normals' / f'{i:06d}.normal.png')
    print('Evaluating Done!')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # set data / training arguments
    parser.add_argument('--model_checkpoint_file', help="Path to model checkpoint file")
    parser.add_argument('--grid_checkpoint_file', help="Path to model checkpoint file")
    parser.add_argument('--output_path', help="Path to output")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--full', default='True', type=eval, choices=[True, False])

    args = parser.parse_args()
    main(args)