import sys
sys.path.append('.')
sys.path.append('..')

import torch
import numpy as np
from src.utils.checkpoints import Checkpointer
import math
from pathlib import Path
from tqdm import tqdm
import json
from src.viewer.viewer import Viewer
from src.viewer.camera import Camera

def main(args):
    model_ckpt = torch.load(args.model_checkpoint_file, args.device)
    model = model_ckpt['class'](**model_ckpt['arg_dict'])
    model.load_state_dict(model_ckpt['state_dict'])
    model = model.eval().to(args.device)

    grid_ckpt = torch.load(args.grid_checkpoint_file, args.device)
    grid = grid_ckpt['class'](**grid_ckpt['arg_dict'])
    grid.poses = grid_ckpt['poses']
    grid.load_state_dict(grid_ckpt['state_dict'])
    grid = grid.eval().to(args.device)

    camera = Camera(400, 400, 45 / 180.0 * math.pi, 1, device=args.device)
    if grid.poses is not None:
        pose = grid.poses[0]
        camera.from_pose(pose.cpu().numpy())
    viewer = Viewer(camera, model, grid)
    
    viewer.loop()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # set data / training arguments
    parser.add_argument('--model_checkpoint_file', help="Path to model checkpoint file")
    parser.add_argument('--grid_checkpoint_file', help="Path to model checkpoint file")
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    main(args)