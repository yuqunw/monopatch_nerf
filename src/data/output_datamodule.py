import json
import torch
from pathlib import Path
from .datasets import *
from .patch_datasets import *
import numpy as np
import imageio as iio
import math

class DataModule:
    """
    Dataset for loading transforms.json file 

    the transforms.json must be formatted following way:
    {
        "w": width of images,
        "h": height of images,
        "camera_angle_x": horizontal field of view
        "fl_x": focal length in x, optional if 'fov' is provided
        "fl_y": focal length in y, optional if 'fov' is provided
        "cx": camera center x in pixels, optional if 'fov' is provided
        "cy": camera center y in pixels, optional if 'fov' is provided
        "aabb_scale": scale of the scene
        "global_trans": 4x4 translation matrix mapping scene to original coordinates
        "frames": [
            {
                "transform_matrix": [16 (4x4) opengl coordinates of camera pose],
                "file_path": relative path to image file from transforms.json
                "normal_file_path": relative path to image file from transforms.json
                "depth_file_path": relative path to image file from transforms.json
            }
        ]
    }

    The directory should be formatted as following:
    ROOT/
        images/
            image files
        depths/
            depth files (.pfm)
        normals/
            normal files (.png)
        transforms.json
    """
    def __init__(self, root, batch_size=4096, num_workers=0, width=None, 
                 height=None, val_inv_scale=4, near=0, 
                 test=False, device='cpu'):
        super().__init__()
        self.root = Path(root)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.val_inv_scale = val_inv_scale
        self.near = near # near in 'meters'
        self.device = device
        self.setup_test()


    def setup_test(self):
        camera, samples = self.load_transform(self.root / 'transforms.json')
        center, scale = self.load_center_scale(self.root / 'transforms.json')
        self.dataset = ImageDataset(samples, camera, self.device)
        self.camera = camera
        self.center, self.scale = center, scale

    def load_center_scale(self, transform_path):
        with open(transform_path, 'r') as f:
            transforms = json.load(f)
        center = torch.Tensor(transforms['pose_offset']).view(3)
        scale = transforms['pose_scale']
        return center, scale

    def load_transform(self, transform_path):
        with open(transform_path, 'r') as f:
            transforms = json.load(f)

        samples = []
        frames = transforms['frames']
        if 'aabb_scale' in transforms:
            self.scale = transforms['aabb_scale']
        else:
            self.scale = 1.0
        scale_mat = torch.eye(4) * self.scale
        scale_mat[3, 3] = 1
        if 'global_trans' in transforms:
            self.trans = torch.Tensor(transforms['global_trans']).view(4, 4) @ scale_mat
        else:
            self.trans = scale_mat

        camera = {}

        # rescale image appropriately
        if ('w' in transforms) and ('h' in transforms): 
            OW = transforms['w']
            OH = transforms['h']
        else:
            image_filename = frames[0]['file_path']
            if not image_filename.endswith('.png') or image_filename.endswith('.jpg'):
                image_filename = image_filename + '.png'
            image = iio.imread(self.root / image_filename)
            OH, OW = image.shape[:2]

        W = OW
        H = OH

        if self.width is not None:
            W = self.width
        else:
            self.width = W


        if self.height is not None:
            H = self.height
        else:
            self.height = H

        self.val_height = self.height // self.val_inv_scale
        self.val_width = self.width // self.val_inv_scale

        camera['w'] = W
        camera['h'] = H
        sx = float(W) / float(OW)
        sy = float(H) / float(OH)

        if all([(key in transforms) for key in ['fl_x', 'fl_y', 'cx', 'cy']]):
            K = torch.eye(3)
            K[0, 0] = transforms['fl_x'] * sx
            K[1, 1] = transforms['fl_y'] * sy
            K[0, 2] = transforms['cx'] * sx
            K[1, 2] = transforms['cy'] * sy
            camera['intrinsic'] = K
            focal_length = K[0, 0]
        else:
            focal_length  = (W / 2.0) / (math.tan(transforms['camera_angle_x']/ 2))

            K = torch.eye(3)
            K[0, 0] = focal_length
            K[1, 1] = focal_length
            K[0, 2] = (W / 2.0)
            K[1, 2] = (H / 2.0)
            camera['intrinsic'] = K

        for frame in frames:
            pose = torch.Tensor(frame['transform_matrix']).view(4, 4)
            image_filename = frame['file_path']
            if not image_filename.endswith('.png') or image_filename.endswith('.jpg'):
                image_filename = image_filename + '.png'
            image_file = self.root / image_filename

            pose[:, 1:3] *= -1
            pose[:3, 3:] /= self.scale

            sample = {
                'image_file': image_file,
                'pose': pose,
                'near': self.near,
            }
            if 'depth_path' in frame:
                depth_file = self.root / frame["depth_path"]
                sample['depth_file'] = depth_file   
            if 'mask_file_path' in frame:
                mask_file = self.root / frame["mask_file_path"]
                sample['mask_file'] = mask_file
            samples.append(sample)

        return camera, samples 