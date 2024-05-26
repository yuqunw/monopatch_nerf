import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from tqdm import tqdm
from src.utils.arg_dict_class import ArgDictClass
import nerfacc
from nerfacc import OccGridEstimator

class NeRFAccSampler(ArgDictClass, nn.Module):
    ''' Holds occupancy grid cache for faster sampling 

    The box bound is between [-1, -1, -1] to [1, 1, 1]
    '''
    def __init__(self, 
        radius=1.0, 
        target_batch_size=2 ** 20,
        target_step_size=2048,
        resolution=128,
        decay_weight=0.95,
        update_interval=16,
        warmup_steps=256,
        density_query=lambda x: x,
    ):
        super().__init__()
        self.radius = radius
        self.resolution = resolution
        self.decay =decay_weight
        self.update_interval = update_interval
        self.warmup_steps = warmup_steps
        r = radius
        roi_aabb = [-r, -r, -r, r, r, r]
        self.aabb = nn.Parameter(torch.tensor(roi_aabb), False)
        self.grid = OccGridEstimator(roi_aabb, resolution=resolution, levels=1)
        self.density_query = density_query
        self.target_batch_size = target_batch_size
        self.target_step_size = target_step_size
        self.render_step_size = (radius * 2) * math.sqrt(3) / self.target_step_size
        self.density_culls = nn.Parameter(torch.ones_like(self.grid.binaries), False)
        self.poses = None
        self.camera = None
        
    def init(self):
        return

    def update(self, step):
        self.step = step
        occ_thre = 0.01
        def occ_eval(x):
            density = self.density_query(x)
            occ = 1 - (-NF.relu(density) * self.render_step_size).exp()
            return occ
        self.grid.update_every_n_steps(step, occ_eval,occ_thre, ema_decay=self.decay, warmup_steps=self.warmup_steps, n=self.update_interval)
        # update 
        self.grid.binaries &= self.density_culls

    def sample(self, rays_o, rays_d):
        def sigma_fn(t_starts, t_ends, ray_i):
            t_origins = rays_o[ray_i]
            t_dirs = rays_d[ray_i]
            points = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            return self.density_query(points)[..., 0]

        ray_indices, t_starts, t_ends = self.grid.sampling(
            rays_o, rays_d,
            sigma_fn=sigma_fn,
            render_step_size = self.render_step_size,
            stratified = self.training,
            cone_angle = 0.0,
            alpha_thre=0.0
        )
        return ray_indices, t_starts, t_ends

    def sample_random_points(self, num_points):
        points = torch
        return points


    def cull(self, datamodule, culling_thre=0):
        '''
        Given poses of shape Nx4x4, cull grids
        '''
        self.camera = datamodule.camera
        self.poses = datamodule.poses
        dataset = datamodule.train_image_dataset

        dev = self.grid.grid_coords.device
        ray = dataset.ray.to(dev)
        cell_width = 2 * self.radius / self.grid.resolution[0]
        cell_centers = (self.grid.grid_coords + 0.5) * cell_width - 1
        density_culls = torch.zeros_like(self.grid.binaries, dtype=int).to(dev)
        K = self.camera['intrinsic'].to(dev)
        W = self.camera['w']
        H = self.camera['h']
        print('Culling!', flush=True)
        for batch in dataset:
            pose = batch['pose']
            depth = batch['d']
            # mask = batch['m']
            position = pose[:3, 3].view(1, 3)
            orientation = pose[:3, :3].view(3, 3)
            img_centers = ((cell_centers - position) @ orientation) @ K.T
            xys = img_centers[:, :2] / img_centers[:, 2:]
            x = xys[:, 0]
            y = xys[:, 1]
            z = img_centers[:, 2]
            
            # check for inclusion
            x_valids = (x >= 0) & (x < W)
            y_valids = (y >= 0) & (y < H)
            grid_xy = torch.stack([(x - W / 2.0) / (W / 2.0), (y - H/ 2.0) / (H / 2.0)], dim=-1)
            proj_depth = nn.functional.grid_sample(depth.view(1, 1, H, W), grid_xy.view(1, 1, -1, 2), align_corners=True, mode='bilinear').view(*z.shape)
            # proj_mask = nn.functional.grid_sample(mask.view(1, 1, H, W), grid_xy.view(1, 1, -1, 2), align_corners=True, mode='bilinear').view(*z.shape)
            z_valids = (z > proj_depth * 0.80) & (z < proj_depth * 1.20)# & (proj_mask == 1.0)

            all_valids = (x_valids & y_valids & z_valids).view(density_culls.shape)
            density_culls += all_valids
        self.density_culls.copy_(density_culls > culling_thre)
