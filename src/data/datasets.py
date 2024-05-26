import torch
from torch.utils.data import IterableDataset, Dataset
from .data_utils import get_rays, load_sample, load_sample_eval
from tqdm import tqdm
import random

class ImageSampleDataset(IterableDataset):
    def __init__(self, samples, camera, batch_size=4096, target_batch_size=2**20, device='cpu'):
        super().__init__()
        self.width = camera['w']
        self.height = camera['h']
        # self.fov = camera['fov']
        self.K = None
        if 'intrinsic' in camera:
            self.K = camera['intrinsic']
        self.ray, self.ray_scale = get_rays(self.height, self.width, camera)

        self.samples = samples
        self.batch_size = batch_size
        self.target_batch_size = target_batch_size
        self.samples_per_batch = target_batch_size / batch_size
        self.num_images = len(self.samples)

        self.device = device

    @property
    def num_batches(self):
        return min(max(int(self.target_batch_size / self.samples_per_batch), 1), 1024 * 16)

    def __iter__(self):
        # random sample
        return self

    def __next__(self):
        index = random.randint(0, len(self.samples)-1)
        sample = self.samples[index]
        pose = sample['pose']
        data = load_sample(self.ray, self.ray_scale, self.height, self.width, sample)

        num_rays = data['rgb'].shape[0]
        rand_inds = torch.randperm(num_rays)[:self.num_batches]
        data = self.select_by_inds(data, rand_inds)
        data['pose'] = pose
        data['inds'] = rand_inds
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                data[k] = v
        return data

    def select_by_inds(self, data_dict, inds):
        ret_dict = {}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                ret_dict[k] = v[inds]
            else:
                ret_dict[k] = v
        return ret_dict


class ImageCachedDataset(IterableDataset):
    def __init__(self, samples, camera, batch_size=4096, target_batch_size=2**20, device='cpu'):
        super().__init__()
        self.width = camera['w']
        self.height = camera['h']
        # self.fov = camera['fov']
        self.K = None
        if 'intrinsic' in camera:
            self.K = camera['intrinsic']
        self.ray, self.ray_scale  = get_rays(self.height, self.width, camera)

        self.samples = samples
        self.batch_size = batch_size
        self.target_batch_size = target_batch_size
        self.samples_per_batch = target_batch_size / batch_size
        self.num_rays = batch_size
        self.num_splits = 4
        self.num_images = len(self.samples)
        self.num_image_samples = len(self.samples)

        self.device = device

        self.cache = []

        print('Loading samples!', flush=True)
        for sample in self.samples:
            self.cache.append(load_sample(self.ray, self.ray_scale, self.height, self.width, sample))

    @property
    def num_batches(self):
        return min(max(int(self.target_batch_size / self.samples_per_batch), 1), 1024 * 16)

    def __iter__(self):
        # random sample
        return self

    def __next__(self):
        index = random.randint(0, len(self.samples)-1)
        sample = self.samples[index]
        pose = sample['pose']
        data = self.cache[index]

        num_rays = data['rgb'].shape[0]
        rand_inds = torch.randperm(num_rays)[:self.num_batches]
        data = self.select_by_inds(data, rand_inds)
        data['pose'] = pose
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                data[k] = v
        return data

    def select_by_inds(self, data_dict, inds):
        ret_dict = {}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                ret_dict[k] = v[inds]
            else:
                ret_dict[k] = v
        return ret_dict


class ImageDataset(Dataset):
    def __init__(self, samples, camera, device='cpu'):
        super().__init__()
        self.width = camera['w']
        self.height = camera['h']
        # self.fov = camera['fov']
        self.K = None
        if 'intrinsic' in camera:
            self.K = camera['intrinsic']
        self.ray, self.ray_scale = get_rays(self.height, self.width, camera)

        self.samples = samples
        self.device = device

    def __len__(self):
        # random sample
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        pose = sample['pose']
        data = load_sample(self.ray, self.ray_scale, self.height, self.width, sample)
        data['pose'] = pose
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                data[k] = v
        return data
    
class ImageEvalDataset(Dataset):
    def __init__(self, samples, camera, device='cpu'):
        super().__init__()
        self.width = camera['w']
        self.height = camera['h']
        # self.fov = camera['fov']
        self.K = None
        if 'intrinsic' in camera:
            self.K = camera['intrinsic']
        self.ray, self.ray_scale = get_rays(self.height, self.width, camera)

        self.samples = samples
        self.device = device

    def __len__(self):
        # random sample
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        pose = sample['pose']
        data = load_sample_eval(self.ray, self.ray_scale, self.height, self.width, sample)
        data['pose'] = pose
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                data[k] = v
        return data    
