import torch
from torch.utils.data import IterableDataset, Dataset
from .data_utils import get_rays, load_sample
from tqdm import tqdm
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

class PatchSampleDataset(IterableDataset):
    def __init__(self, samples, camera, batch_size=4096, target_batch_size=2**20, patch_size=16, device='cpu'):
        super().__init__()
        self.width = camera['w']
        self.height = camera['h']
        self.patch_size = patch_size
        # self.fov = camera['fov']
        self.K = None
        if 'intrinsic' in camera:
            self.K = camera['intrinsic']
        self.ray, self.ray_scale  = get_rays(self.height, self.width, camera)

        self.samples = samples
        self.batch_size = min(max((batch_size / (patch_size ** 2)), 1), 128)
        self.target_batch_size = target_batch_size
        self.samples_per_batch = target_batch_size / self.batch_size
        self.num_images = len(self.samples)
        self.num_image_samples = len(self.samples)
        self.device = device

    @property
    def num_patches(self):
        return min(max(int(self.target_batch_size / self.samples_per_batch), 1), 128)

    def __iter__(self):
        # random sample
        return self

    def __next__(self):

        index = random.randint(0, len(self.samples)-1)
        sample = self.samples[index]
        pose = sample['pose']
        data = load_sample(self.ray, self.ray_scale, self.height, self.width, sample)
        data = self.sample_patches(data, self.num_patches)
        data['pose'] = pose
        data['K'] = self.K

        # add random origins
        random_t = torch.rand_like(data['ray_o'][:, :1, :1])
        random_t -= 0.5
        random_t *= 0.1
        data['rand_o'] = data['ray_o'] + random_t 

        # data['inds'] = rand_inds
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                data[k] = v
        return data


    def sample_patches(self, data_dict, num_samples):
        ret_dict = {}

        # Sample start locations
        x0 = torch.randint(0, self.width - self.patch_size + 1, size=(num_samples, 1, 1))
        y0 = torch.randint(0, self.height - self.patch_size + 1, size=(num_samples, 1, 1))
        xy0 = torch.cat([x0, y0], axis=-1)
        patch_idx = xy0 + torch.stack(
            torch.meshgrid(torch.arange(self.patch_size), torch.arange(self.patch_size), indexing='xy'),
            axis=-1).view(1, -1, 2)
        inds = (patch_idx[..., 0] + patch_idx[..., 1] * self.width).view(-1)

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                ret_dict[k] = v[inds].view(num_samples, self.patch_size, self.patch_size, -1)
            else:
                ret_dict[k] = v
        return ret_dict

class PatchCachedDataset(IterableDataset):
    def __init__(self, samples, camera, batch_size=4096, target_batch_size=2**20, patch_size=16, device='cpu'):
        super().__init__()
        self.width = camera['w']
        self.height = camera['h']
        self.patch_size = patch_size
        # self.fov = camera['fov']
        self.K = None
        if 'intrinsic' in camera:
            self.K = camera['intrinsic']
        self.ray, self.ray_scale  = get_rays(self.height, self.width, camera)
        self.samples = samples
        self.batch_size = min(max((batch_size / (patch_size ** 2)), 1), 128)
        self.target_batch_size = target_batch_size
        self.samples_per_batch = target_batch_size / self.batch_size
        self.num_images = len(self.samples)
        self.num_image_samples = len(self.samples)

        self.device = device

        self.cache = []

        print('Loading samples!', flush=True)
        for sample in tqdm(self.samples, total=len(self.samples)):
            self.cache.append(load_sample(self.ray, self.ray_scale, self.height, self.width, sample))

    @property
    def num_patches(self):
        return min(max(int(self.target_batch_size / self.samples_per_batch), 1), 128)

    def __iter__(self):
        # random sample
        return self

    def __next__(self):
        index = random.randint(0, len(self.samples)-1)
        sample = self.samples[index]
        pose = sample['pose']
        data = self.cache[index]
        data = self.sample_patches(data, self.num_patches)
        data['pose'] = pose
        data['K'] = self.K

        # add random origins
        random_t = torch.rand_like(data['ray_o'][:, :1, :1])
        random_t -= 0.5
        random_t *= 0.1
        data['rand_o'] = data['ray_o'] + random_t 

        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                data[k] = v
        return data
    
    def sample_patches(self, data_dict, num_samples):
        ret_dict = {}

        # Sample start locations
        x0 = torch.randint(0, self.width - self.patch_size + 1, size=(num_samples, 1, 1))
        y0 = torch.randint(0, self.height - self.patch_size + 1, size=(num_samples, 1, 1))
        xy0 = torch.cat([x0, y0], axis=-1)
        patch_idx = xy0 + torch.stack(
            torch.meshgrid(torch.arange(self.patch_size), torch.arange(self.patch_size), indexing='xy'),
            axis=-1).view(1, -1, 2)
        inds = (patch_idx[..., 0] + patch_idx[..., 1] * self.width).view(-1)

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                ret_dict[k] = v[inds].view(num_samples, self.patch_size, self.patch_size, -1)
            else:
                ret_dict[k] = v
        return ret_dict

