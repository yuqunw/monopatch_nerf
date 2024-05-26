import sys
sys.path.append('.')
sys.path.append('..')

import torch
from src.data.output_datamodule import DataModule
import math
from pathlib import Path
from tqdm import tqdm
from src.utils.image import *
from pyntcloud import PyntCloud
import torch.nn.functional as NF
import pandas as pd
from src.utils.colmap import read_model
import time
import open3d as o3d
import colorsys

def get_distinct_colors(n_clusters):
    colors = []
    for i in range(n_clusters):
        # Divide the hue range into equal parts
        hue = i / n_clusters
        # Convert HSV color to RGB
        rgb_color = colorsys.hsv_to_rgb(hue, 1, 1) # Keeping saturation and value constant
        # Convert RGB from [0, 1] range to [0, 255] range
        rgb_color = tuple(int(c * 255) for c in rgb_color)
        colors.append(rgb_color)
    return np.array(colors)


def main(args):
    check_consistency(args.output_path, args.device)


def check_consistency(output_path, device):
    datamodule = DataModule(output_path, device=device)
    scale = datamodule.scale
    offset = datamodule.center
    dataset = datamodule.dataset
    num_images = len(dataset)
    H, W = datamodule.height, datamodule.width
    camera = datamodule.camera
    ray = dataset.ray.to(device)
    
    # iterate and write to depth
    ones = torch.ones_like(ray.view(-1, 3)[..., :1])
    all_world_points = []
    all_world_colors = []
    print('loading data...')
    poses = []
    depths = []
    rgbs = []
    normals = []
    H, W = 640, 960
    for index, batch in enumerate(dataset):
        if index >= args.check_num:
            break
        poses.append(batch['pose'])
        depths.append(batch['d'].view(H, W, -1))
        rgbs.append(batch['rgb'].view(H, W, -1))
        normals.append(batch['n'].view(H, W, -1))
    poses = torch.stack(poses)
    depths = torch.stack(depths)
    rgbs = torch.stack(rgbs)
    normals = torch.stack(normals)

    K = camera['intrinsic']

    # random_colors = torch.rand((args.check_num, 3), device=device)
    random_colors = torch.tensor(get_distinct_colors(args.check_num)).to(device) / 255.
    color_list = []
    point_list = []
    for i in range(min(args.check_num, len(dataset))):
        pose = poses[i]
        depth = depths[i]
        color = random_colors[i]
        # project 
        norm_rays = ray.view(-1, 3) / ray.view(-1, 3)[..., 2:]
        points = norm_rays * depth.view(-1, 1)
        h_points = torch.cat((points, ones), -1)

        # transform using pose to world coordinate
        world_hpoints = (pose @ h_points.t()).t() # (H*W, 4)

        point_list.append(world_hpoints[:, :3]) # N, 3
        color_list.append(torch.ones_like(world_hpoints[:, :3]) * color) # N, 3
    
    points = (torch.cat(point_list, 0).cpu() * scale + offset).numpy()
    colors = torch.cat(color_list, 0).cpu().numpy()
    # points = points * scale + offset

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])
    # Save point cloud
    o3d.io.write_point_cloud("consistency_da.ply", pcd)
    # 


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # set data / training arguments
    parser.add_argument('--output_path', help="Path to output")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--check_num', default=37, type=int)

    args = parser.parse_args()
    main(args)