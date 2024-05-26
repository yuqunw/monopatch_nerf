import math
import numpy as np
import torch
import torch.nn.functional as NF

class Camera:
    def __init__(self, width, height, fov_x, radius=1, pan_speed=0.1, zoom_speed=1.1, rot_speed=1, device='cuda') -> None:
        self.width = width
        self.height = height 
        self.fov_x = fov_x
        self.fl = self.width / math.tan(self.fov_x / 2.) / 2 
        self.device = device

        # up direction refers to the z+
        self.up = np.zeros(3)
        self.up[2] = 1

        self.theta = 0
        self.phi = math.pi / 4
        self.radius = 0.25 * radius
        self.max_bound = radius

        self.target = np.zeros(3) + 0.5
        self.pan_speed = pan_speed
        self.zoom_speed = zoom_speed
        self.rot_speed = rot_speed

        self.ray_cache = self.create_ray_cache().to(self.device)

    def normalize(self, v):
        return v / np.linalg.norm(v, ord=2)

    def zoom(self, v):
        self.radius *= self.zoom_speed ** v
        self.radius = max(self.radius, 1e-3)
        self.radius = min(self.radius, self.max_bound * 3)

    def rotate(self, dx, dy):
        self.theta += dx * self.rot_speed
        self.phi -= dy * self.rot_speed
        self.theta %= 2 * math.pi
        self.phi = min(math.pi - 5e-2, self.phi)
        self.phi = max(5e-2, self.phi)

    def pan(self, dx, dy, dz=0):
        z = self.normalize(self.target - self.position)
        x = self.normalize(np.cross(z, self.up))
        y = np.cross(x, z)

        # handle panning up
        self.target -= y * dy * self.pan_speed

        # handle panning right
        self.target += x * dx * self.pan_speed

        # handle panning forward
        self.target += z * dz * self.pan_speed

    @property
    def position(self):
        cam_dir = np.zeros(3)
        cam_dir[0] = math.sin(self.phi) * math.sin(self.theta)
        cam_dir[1] = math.sin(self.phi) * math.cos(self.theta)
        cam_dir[2] = math.cos(self.phi)
        return (cam_dir * self.radius) + self.target

    @property
    def pose(self):
        z = self.normalize(self.target - self.position)
        x = self.normalize(np.cross(z, self.up))
        y = np.cross(x, z)

        P = np.eye(4)

        P[:3, 3] = self.position

        P[:3, 0] = x
        P[:3, 1] = y
        P[:3, 2] = z
        return P

    def from_pose(self, pose):
        t = 0.1
        position = pose[:3, 3]
        Rt = pose[:3, :3]
        # x+
        right = Rt[:, 0]
        # y+
        up = Rt[:, 1]
        # z+
        lookat = Rt[:, 2]

        self.target = position + lookat * t
        self.up = up

        self.phi = math.acos(-lookat[2])
        stheta = -lookat[0] / math.sin(self.phi)
        ctheta = -lookat[1] / math.sin(self.phi)
        self.theta = math.atan2(stheta, ctheta)

        self.radius = t


    def create_ray_cache(self):
        w = self.width
        h = self.height
        f = self.fl
        O = 0.5
        cx = w / 2.
        cy = h / 2.
        x_coords = np.linspace(O, w - 1 + O, w)
        y_coords = np.linspace(O, h - 1 + O, h)

        # HxW grids
        x_grid_coords, y_grid_coords = np.meshgrid(x_coords, y_coords)

        # HxWx3
        ray = np.stack([
            (x_grid_coords - cx) / f,
            (y_grid_coords - cy) / f,
            np.ones_like(x_grid_coords) 
        ], axis=-1)

        # ray /= np.linalg.norm(ray, ord=2, axis=-1, keepdims=True)

        return NF.normalize(torch.from_numpy(ray).float(), p=2, dim=-1)

    def get_rays(self):
        ray = self.ray_cache
        pose = torch.from_numpy(self.pose).to(ray).float()
        ray_d = ray @ pose[:3, :3].T
        ray_o = pose[:3, 3].view(1, 1, 3).expand(ray_d.shape)
        H, W = ray_o.shape[:2]
        ray_o = ray_o.view(-1, 3)
        ray_d = ray_d.view(-1, 3)
        near, far, mask = self.compute_near_far(ray_o, ray_d)
        ray_o = ray_o.view(H, W, 3)
        ray_d = ray_d.view(H, W, 3)
        near = near.view(H, W, 1)
        far = far.view(H, W, 1)
        mask = mask.view(H, W, 1)
        return torch.cat((ray_o, ray_d, near, far, mask.float()), -1)

    def compute_near_far(self, rays_o, rays_d):
        # t = (0 - O) / D
        # t = (1 - O) / D
        tzero = (-self.max_bound -rays_o) / (rays_d + 1e-15) # [B, 3]
        tone = (self.max_bound - rays_o) / (rays_d + 1e-15)
        ts = torch.stack((tzero, tone), 1) # [B, 2, 3]
        tmins = ts.min(1)[0]
        tmaxs = ts.max(1)[0]

        near = tmins.max(1)[0]
        far = tmaxs.min(1)[0]

        near = torch.clamp(near, min=0.05)
        mask = (far > near) 
        return near, far, mask
