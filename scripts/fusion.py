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

def main(args):
    (Path(args.output_path) / 'results').mkdir(exist_ok=True, parents=True)
    fuse_reconstruction(Path(args.output_path), Path(args.output_path) / 'results' / 'fused.ply', args.threshold, args.min_views, args.device)


def fuse_reconstruction(output_path, ply_file, threshold, min_views, device, sparse_path=None):
    datamodule = DataModule(output_path, device=device)
    scale = datamodule.scale
    offset = datamodule.center
    dataset = datamodule.dataset
    num_images = len(dataset)
    if sparse_path is not None:
        col_camera, col_image, col_points = read_model(str(sparse_path))
        image_ids = sorted(col_image.keys(), key=lambda x: col_image[x].name)
        iid_to_points = {}

        oid = 0
        for col_id in image_ids:
            image = col_image[col_id]
            iid_to_points[oid] = set(i for i in image.point3D_ids if i  != -1)
            oid += 1
            
        pairs = {}
        for i in tqdm(range(num_images), desc='Checking pairs', leave=False, dynamic_ncols=True):
            pairs[i] = {}
            for j in range(num_images):
                if i == j:
                    pairs[i][j] = False
                else:
                    # check for pairs by checking at least 1 overlapping points in the sparse reconstruction
                    image_i_pts = iid_to_points[i]
                    image_j_pts = iid_to_points[j]
                    if len(image_i_pts & image_j_pts) > (len(image_i_pts) * 0.1):
                        pairs[i][j] = True
                    else:
                        pairs[i][j] = False

    else:
        pairs = {}
        for i in range(num_images):
            pairs[i] = {}
            for j in range(num_images):
                if i == j:
                    pairs[i][j] = False
                else:
                    pairs[i][j] = True
    num_pairs = sum([sum([1 for i in v.values() if i]) for k, v in pairs.items()])
    np_num_pairs = np.array([sum([1 for i in v.values() if i]) for k, v in pairs.items()]).astype(np.float32)
    avg_num_pairs = np_num_pairs.mean()
    min_num_pairs = np_num_pairs.min().astype(np.int32)
    max_num_pairs = np_num_pairs.max().astype(np.int32)
    print(f"Running for {num_pairs} / {(num_images - 1) * num_images} (min/max/avg: {min_num_pairs:d}/{max_num_pairs:d}/{avg_num_pairs:.02f}) pairs")
    H, W = datamodule.height, datamodule.width
    fp_16 = False
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
    image_masks = [] # Mask based on semantic segmentation
    for batch in dataset:
        poses.append(batch['pose'])
        depths.append(batch['d'])
        rgbs.append(batch['rgb'])
        image_masks.append(batch['m'])
    poses = torch.stack(poses)
    depths = torch.stack(depths)
    rgbs = torch.stack(rgbs)
    image_masks = torch.stack(image_masks)
    masks = torch.zeros_like(depths).view(num_images, -1) # Mask for fusion
    K = camera['intrinsic'].to(device)
    grid_intrinsic = torch.tensor([
        2.0 / W, 0, -1,
        0, 2.0 / H, -1
    ]).reshape(2, 3).to(device)

    print('starting fusion...')
    start_time = time.time()
    for i in range(num_images):
        # if i == 16:
        #     print('here')
        pose = poses[i]
        depth = depths[i]
        image_mask = image_masks[i]
        # project 
        norm_rays = ray.view(-1, 3) / ray.view(-1, 3)[..., 2:]
        points = norm_rays * depth.view(-1, 1)
        h_points = torch.cat((points, ones), -1)

        # transform using pose to world coordinate
        world_hpoints = (pose @ h_points.t()).t()
        mask_sum = torch.zeros_like(depth).view(-1)
        mask_sum_insde_frame = torch.zeros_like(depth).view(-1)
        m = torch.ones_like(depth).view(-1).bool()

        for j, should_run in pairs[i].items():
            if not should_run:
                continue

            src_pose = dataset[j]['pose']
            src_mask = image_masks[j]
            src_points = (src_pose.inverse() @ world_hpoints.t()).t()
            src_image_hpoints = (K @ src_points[:, :3].t()).t()
            src_image_points = src_image_hpoints / src_image_hpoints[:, 2:]
            src_depth = depths[j]
            src_grid_coords = (grid_intrinsic @ src_image_points.t()).t()
            src_proj_depths = NF.grid_sample(src_depth.view(1, 1, H, W), src_grid_coords.view(1, H, W, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
            src_dists = src_image_hpoints[:, 2]
            baseline = ((pose[:3, 3] - src_pose[:3, 3]) ** 2).sum() ** 0.5
            f = K[0, 0]
            src_disp = f * baseline / src_proj_depths.view(-1)
            # ref_disp = f * baseline / src_image_hpoints[:, 2:].view(-1)
            ref_disp = f * baseline / src_dists.view(-1)
            disp_mask = (ref_disp - src_disp).abs() < threshold
            grid_mask = ((src_grid_coords >= -1) & (src_grid_coords <= 1)).all(-1).view(-1)
            mask = disp_mask & grid_mask & (masks[i].view(-1) == 0) & (src_mask.view(-1) == 1)
            mask_sum += mask.float()
            mask_sum_insde_frame += (~grid_mask).float()
        # m = (mask_sum >= min_views) | (mask_sum_insde_frame == 0) # Make sure that points not visible in any other frames are fused
        m = (mask_sum >= min_views) & (image_mask.view(-1) == 1)

        # iterate and assign masks
        for j, should_run in pairs[i].items():
            if not should_run:
                continue                
            src_pose = dataset[j]['pose']
            src_mask = image_masks[j]
            src_points = (src_pose.inverse() @ world_hpoints.t()).t()
            src_image_hpoints = (K @ src_points[:, :3].t()).t()
            src_image_points = src_image_hpoints / src_image_hpoints[:, 2:] # H * W, 2
            src_depth = depths[j]
            src_grid_coords = (grid_intrinsic @ src_image_points.t()).t()
            src_proj_depths = NF.grid_sample(src_depth.view(1, 1, H, W), src_grid_coords.view(1, H, W, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
            src_dists = src_image_hpoints[:, 2]
            baseline = ((pose[:3, 3] - src_pose[:3, 3]) ** 2).sum() ** 0.5
            f = K[0, 0]
            src_disp = f * baseline / src_proj_depths.view(-1)
            # ref_disp = f * baseline / src_image_hpoints[:, 2:].view(-1)
            ref_disp = f * baseline / src_dists.view(-1)
            disp_mask = (ref_disp - src_disp).abs() < threshold       

            src_image_id = src_image_points[:, :2].floor().int() # H * W, 2
            src_image_mask = (src_image_id[:, 0] >= 0) & (src_image_id[:, 0] < W) & (src_image_id[:, 1] >= 0) & \
                             (src_image_id[:, 1] < H) & (src_image_hpoints[:, 2] > 0) & (src_mask.view(-1) == 1)
            mask_to_note = (src_image_mask & m & disp_mask)
            if mask_to_note.sum() != 0:
                src_image_mask_ids = src_image_id[mask_to_note]
                src_image_flat_ids = (src_image_mask_ids[:, 0] + src_image_mask_ids[:, 1] * W).unique().long()
                masks[j][src_image_flat_ids] += 1      

        world_points = world_hpoints[:, :3][m]
        world_colors = rgbs[i].view(-1, 3)[m]

        all_world_points.append(world_points)
        all_world_colors.append(world_colors)
        curr_time = time.time()
        elapsed_time = curr_time - start_time
        eta = elapsed_time / (i + 1) * (num_images - i - 1)
        eta_hours = math.floor(eta / 3600)
        eta_minutes = math.floor((eta - eta_hours * 3600) / 60)
        eta_seconds = math.floor(eta - eta_hours * 3600 - eta_minutes * 60)
        print(f'Fused image {i+1}/{num_images}, eta {eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}', flush=True)

    all_world_points = torch.cat(all_world_points) * scale + offset.view(1, 3).to(device)
    all_world_colors = torch.cat(all_world_colors)
    all_world_points = all_world_points.cpu().numpy()
    all_world_colors = (all_world_colors * 255).cpu().numpy().astype(np.uint8)
    print(f'Creating point cloud', flush=True)
    pointdata = pd.DataFrame({
        'x': all_world_points[:, 0],
        'y': all_world_points[:, 1],
        'z': all_world_points[:, 2],
        'red': all_world_colors[:, 0],
        'green': all_world_colors[:, 1],
        'blue': all_world_colors[:, 2],
    })
    pointcloud = PyntCloud(pointdata)
    pointcloud.to_file(str(ply_file))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # set data / training arguments
    parser.add_argument('--output_path', help="Path to output")
    parser.add_argument('--min_views', default=2, type=int, help="Path to output")
    parser.add_argument('--threshold', default=2.0, type=float, help="Path to output")
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    main(args)
