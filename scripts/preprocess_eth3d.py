import sys
sys.path.append('..')
sys.path.append('.')
import torch
import torch.nn.functional as NF
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from src.utils.colmap import read_model, qvec2rotmat
from omnidata import OmnidataDepthModel, OmnidataNormalModel
from ade20k_segmenter import ADE20KSegmenter
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def scale_intrinsics(K, OH, OW, NH, NW):
    K = K.copy()
    K[0] *= NW / float(OW)
    K[1] *= NH / float(OH)
    return K

def find_transforms_center_and_scale(raw_transforms):
    fx = raw_transforms['fl_x']
    fy = raw_transforms['fl_y']
    cx = raw_transforms['cx']
    cy = raw_transforms['cy']
    w = raw_transforms['w']
    h = raw_transforms['h']
    yz_neg = np.array([
        1, 0, 0,
        0, -1, 0,
        0, 0, -1,
    ]).reshape(3, 3)
    K = np.array([
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1
    ]).reshape(3, 3) @ yz_neg

    image_corners = np.array([
        0.5, 0.5, 1,
        w-0.5, 0.5, 1,
        0.5, h-0.5, 1,
        w-0.5, h-0.5, 1,
    ]).reshape(-1, 3)

    corner_points = image_corners @ np.linalg.inv(K).T


    frames = raw_transforms['frames']
    world_points = []
    for frame in tqdm(frames, desc="Computing Optimal AABB"):
        pose = np.array(frame['transform_matrix'])
        far = frame['far']
        if far != 0:

            proj_corner_points = corner_points * far
            proj_corner_hpoints = np.concatenate([
                proj_corner_points,
                np.ones_like(proj_corner_points[:, :1])
            ], 1)
            world_point = proj_corner_hpoints @ pose.T
            world_points.append(world_point[:, :3])
    # compute aabb 
    world_points = np.concatenate(world_points)
    mins = world_points.min(0)
    maxs = world_points.max(0)
    center = (mins + maxs) / 2.0
    scale = (maxs - mins).max() / 2.0
    return center, scale

def parse_camera(camera):
    if camera.model == 'SIMPLE_PINHOLE':
        f, cx, cy = camera.params
        fx = f
        fy = f
    elif camera.model == 'PINHOLE':
        fx, fy, cx, cy = camera.params

    return fx, fy, cx, cy

def main(args):
    '''
    Preprocesses input folder (containing colmap files / undistorted images) 
    to output folder (containing transforms.json / images / normals)

    sparse_folder:
        images.txt
        points3D.txt
        cameras.txt
    input_image_folder:
        should contain images with the paths in images.txt
        e.g.)
        with images.txt's image.name as 'images/000000.png'

        images/
            000000.png

    output_folder:
        images/
        transforms.json
    '''
    input_path = Path(args.input_image_folder)

    cameras, images, points = read_model(args.sparse_folder)

    # for now, we have should only have one camera!
    # assert len(cameras) == 1
    num_images = len(images)

    camera = cameras[list(cameras.keys())[0]]
    OW = camera.width
    OH = camera.height
    fx, fy, cx, cy = parse_camera(camera)
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    output_path = Path(args.output_folder)
    (output_path / 'images').mkdir(exist_ok=True, parents=True)
    (output_path / 'segs').mkdir(exist_ok=True, parents=True)
    (output_path / 'depths').mkdir(exist_ok=True, parents=True)
    (output_path / 'met_depths').mkdir(exist_ok=True, parents=True)
    (output_path / 'normals').mkdir(exist_ok=True, parents=True)

    device = torch.device(args.device)
    infer = True
    if infer:
        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  
        repo = "isl-org/ZoeDepth"
        depth_model = OmnidataDepthModel().to(device)
        met_depth_model = torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(device)
        normal_model = OmnidataNormalModel().to(device)
        seg_model = ADE20KSegmenter().to(device)
        depth_model.eval()
        met_depth_model.eval()
        normal_model.eval()
        seg_model.eval()

    if (args.width is not None) and (args.height is not None):
        NW = args.width
        NH = args.height
        K = scale_intrinsics(K, OH, OW, NH, NW)
    else:
        NW = OH
        NH = OW
    sx = NW / float(OW)
    sy = NH / float(OH)

    # misc. trans
    trans =  transforms.Compose([
        transforms.Resize((NH, NW), interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])
    norm_trans = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    center_trans = transforms.Normalize(mean=0.5, std=0.5)

    to_pil = transforms.ToPILImage()

    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    fovx = math.atan2(cx, fx) * 2
    fovy = math.atan2(cy, fy) * 2

    frames = []
    transforms_data = {
        'camera_angle_x': fovx,
        'camera_angle_y': fovy,
        'fl_x': fx,
        'fl_y': fy,
        'k1': 0,
        'k2': 0,
        'p1': 0,
        'p2': 0,
        'cx': cx,
        'cy': cy,
        'w': NW,
        'h': NH,
        'frames': frames,
        "aabb_scale": args.aabb_scale
    }
    poses = []

    image_ids = sorted(images.keys(), key=lambda x: images[x].name)

    # first compute AABB from the sparse point cloud
    valid_pids = set()
    for iid in tqdm(image_ids, leave=False, total=num_images):
        image = images[iid]

        image_file = input_path / image.name
        qvec = image.qvec
        tvec = image.tvec
        R = qvec2rotmat(qvec)
        E = np.eye(4)
        ref_pids = []
        for pid in image.point3D_ids:
            if (pid >= 0) and (pid in points) and (points[pid].error < 1.0):
                ref_pids.append(pid)

        # compute near far
        if len(ref_pids) > 3:
            world_points= np.stack([points[sp_pid].xyz for sp_pid in ref_pids])
            ref_points = torch.from_numpy((R @ world_points.T + tvec.reshape(3, 1)).T)
            ref_sparse_depths = ref_points[:, 2].cpu()
            far = (ref_sparse_depths.quantile(0.95) * 1.1).item()
            valid_pid = [ref_pids[i] for i, is_in in enumerate(ref_sparse_depths < far) if is_in]
            valid_pids |= set(valid_pid)
    valid_points = np.stack([points[pid].xyz for pid in valid_pids])
    mins = valid_points.min(0)
    maxs = valid_points.max(0)
    center = (maxs + mins) / 2.0
    scale = (maxs - mins).max() / 1.8

    # iterate over the images to write files
    oid = 0
    valid_pids = set()
    for iid in tqdm(image_ids, leave=False, total=num_images):
        image = images[iid]

        image_file = input_path / image.name

        qvec = image.qvec
        tvec = image.tvec
        R = qvec2rotmat(qvec)
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = tvec
        pose = np.linalg.inv(E)

        pose[:, 1] *= -1
        pose[:, 2] *= -1
        pose[:3, 3] -= center
        pose[:3, 3] /= scale
        poses.append(pose)

        # crop 
        rgb = Image.open(image_file)
        resized = trans(rgb)
        normalized  = norm_trans(resized)
        centered = center_trans(resized)

        resized = resized[:3].unsqueeze(0).to(device)
        normalized = normalized[:3].unsqueeze(0).to(device)
        centered = centered[:3].unsqueeze(0).to(device)

        # run inferences
        with torch.no_grad():
            depth = depth_model(centered)[0].clamp(0, 1)
            met_depth = met_depth_model.infer(resized)[0, 0]

            normal = normal_model(resized)[0].clamp(0, 1)
            seg = seg_model(normalized)[0]
            _, pred = torch.max(seg, dim=0)
            mask = (pred == 2) | (pred == 12) | (pred == 9)
        out_image_file = output_path / 'images' / f'{oid:06d}.rgb.png'
        out_mask_file = output_path / 'images' / f'dynamic_mask_{oid:06d}.rgb.png'
        out_seg_file = output_path / 'segs' / f'{oid:06d}.seg.png'
        out_depth_file = output_path / 'depths' / f'{oid:06d}.depth.tiff'
        out_met_depth_file = output_path / 'met_depths' / f'{oid:06d}.met_depth.tiff'
        out_normal_file = output_path / 'normals' / f'{oid:06d}.normal.png'
        # find a scale for depth
        pids = [(i, pid) for i, pid in enumerate(image.point3D_ids) if (pid >= 0) and (pid in points) and (len(points[pid].image_ids) > 3)]
        if len(pids) > 0:
            ref_pids = image.point3D_ids[[p[0] for p in pids]]
            world_points= np.stack([points[sp_pid].xyz for sp_pid in ref_pids])
            ref_points = torch.from_numpy(K @ (R @ world_points.T + tvec.reshape(3, 1))).T
            ref_sparse_depths = ref_points[:, 2].cpu()
            ref_xys = ref_points[:, :2] / ref_points[:, 2:3]
            ref_xys[:, 0] /= (NW / 2.0)
            ref_xys[:, 1] /= (NH / 2.0)
            ref_xys[:, 0] -= 1

            ref_xys[:, 1] -= 1

            ref_met_depths = NF.grid_sample(met_depth.view(1, 1, NH, NW), ref_xys.view(1, 1, -1, 2).to(depth), align_corners=True, mode='bilinear').view(-1).cpu()
            best_met_depth_scale = 1
            best_met_depth_shift = 0
            max_res = 0
            num_depth_candidates = len(ref_sparse_depths)

            for iter in tqdm(range(1000), desc="RANSAC", leave=False, dynamic_ncols=True):
                # 1.0 = 10m
                # 0.020 = 20cm
                if num_depth_candidates < 2:
                    depth_inds = np.random.choice(num_depth_candidates, 1, replace=False)
                    sparse_ds = ref_sparse_depths[depth_inds].numpy()
                    inf_ds = ref_met_depths[depth_inds].numpy()
                    met_scale = sparse_ds[0] / inf_ds[0]
                    met_shift = 0
                else:
                    depth_inds = np.random.choice(num_depth_candidates, 2, replace=False)
                    sparse_ds = ref_sparse_depths[depth_inds].numpy()
                    inf_ds = ref_met_depths[depth_inds].numpy()
                    met_scale = sparse_ds[0] / inf_ds[0]
                    met_shift = met_scale * inf_ds[1] - sparse_ds[1]
                inliers = (ref_met_depths * met_scale + met_shift - ref_sparse_depths).abs() < 0.02
                res = inliers.float().sum()
                if res > max_res:
                    max_res = res
                    best_met_depth_scale =met_scale 
                    best_met_depth_shift =met_shift 
            met_depth = (best_met_depth_scale * met_depth + best_met_depth_shift) / scale

            ref_depths = NF.grid_sample(depth.view(1, 1, NH, NW), ref_xys.view(1, 1, -1, 2).to(depth), align_corners=True, mode='bilinear').view(-1).cpu()
            best_depth_scale = 1
            best_depth_shift = 1
            max_res = 0
            num_depth_candidates = len(ref_sparse_depths)
            for iter in tqdm(range(1000), desc="RANSAC", leave=False, dynamic_ncols=True):
                # 1.0 = 10m
                # 0.020 = 20cm
                if num_depth_candidates < 2:
                    depth_inds = np.random.choice(num_depth_candidates, 1, replace=False)
                    sparse_ds = ref_sparse_depths[depth_inds].numpy()
                    inf_ds = ref_depths[depth_inds].numpy()
                    met_scale = sparse_ds[0] / inf_ds[0]
                    met_shift = 0
                else:
                    depth_inds = np.random.choice(num_depth_candidates, 2, replace=False)
                    sparse_ds = ref_sparse_depths[depth_inds].numpy()
                    inf_ds = ref_depths[depth_inds].numpy()
                    met_scale = sparse_ds[0] / inf_ds[0]
                    met_shift = met_scale * inf_ds[1] - sparse_ds[1]
                inliers = (ref_depths * met_scale + met_shift - ref_sparse_depths).abs() < 0.02
                res = inliers.float().sum()
                if res > max_res:
                    max_res = res
                    best_depth_scale = met_scale
                    best_depth_shift = met_shift
            depth = (best_depth_scale * depth + best_depth_shift) / scale

        else:
            print('wtf')


        to_pil(resized[0]).save(out_image_file)
        Image.fromarray((normal.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)).save(out_normal_file)
        cv2.imwrite(str(out_depth_file), depth.cpu().numpy().astype(np.float32))
        cv2.imwrite(str(out_met_depth_file), met_depth.cpu().numpy().astype(np.float32))
        to_pil(mask[None].float().repeat(3, 1, 1)).save(out_mask_file)
        Image.fromarray(pred.cpu().numpy().astype(np.uint8)).save(out_seg_file)

        frame = {
            'file_path': f'images/{oid:06d}.rgb.png',
            'seg_path': f'segs/{oid:06d}.seg.png',
            'depth_path': f'depths/{oid:06d}.depth.tiff',
            'met_depth_path': f'met_depths/{oid:06d}.met_depth.tiff',
            'normal_path': f'normals/{oid:06d}.normal.png',
            'mask_file_path': f'images/dynamic_mask_{oid:06d}.rgb.png',
            'transform_matrix': pose.tolist(),
        }
        oid += 1

        frames.append(frame)
    # valid_points = np.stack([points[pid].xyz for pid in valid_pids])
    # mins = valid_points.min(0)
    # maxs = valid_points.max(0)
    # center = (maxs + mins) / 2.0
    # scale = (maxs - mins).max() / 1.8


    # for oid, frame in tqdm(enumerate(frames), leave=False, desc='Transforming', dynamic_ncols=True):
    #     p = np.array(frame['transform_matrix'])
    #     depth = cv2.imread( str(output_path / frame['depth_path']), cv2.IMREAD_UNCHANGED)
    #     p[:3, 3] -= center
    #     p[:3, 3] /= scale
    #     depth = depth / scale
    #     cv2.imwrite( str(output_path / frame['depth_path']), depth)

        # frame['transform_matrix'] = p.tolist()

    transforms_data['pose_scale'] = scale
    transforms_data['pose_offset'] = center.tolist()
    
    import json
    with open(output_path / 'transforms.json', 'w') as f:
        json.dump(transforms_data, f, indent=4)

    train_transforms_data = transforms_data.copy()
    test_transforms_data = transforms_data.copy()
    train_frames = []
    test_frames = []
    for fid, frame in enumerate(frames):
        if fid % 10 == 0:
            test_frames.append(frame)
        else:
            train_frames.append(frame)

    train_transforms_data['frames'] = train_frames
    test_transforms_data['frames'] = test_frames

    with open(output_path / 'transforms_train.json', 'w') as f:
        json.dump(train_transforms_data, f, indent=4)

    with open(output_path / 'transforms_test.json', 'w') as f:
        json.dump(test_transforms_data, f, indent=4)
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--sparse_folder', type=str, required=True)
    parser.add_argument('-i', '--input_image_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('-a', '--aabb_scale', type=int, default=1, choices=[1, 2, 4, 8, 16])
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)