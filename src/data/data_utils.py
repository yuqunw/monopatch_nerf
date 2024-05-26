from PIL import Image
import math
import torch.nn.functional as NF
import torchvision.transforms.functional as F
import torch
import cv2

def get_rays(h, w, camera):
    O = 0.5
    if 'intrinsic' in camera:
        K = camera['intrinsic']
    elif 'fov' in camera:
        fov = camera['fov']
        cx = w / 2.
        cy = h / 2.
        f = cx / math.tan(fov / 2.0)
        K = torch.eye(3)
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    else:
        raise RuntimeError('Camera Must have either intrinsic or fov')
    f = K[0, 0]

    x_coords = torch.linspace(O, w - 1 + O, w)
    y_coords = torch.linspace(O, h - 1 + O, h)

    # HxW grids
    y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])

    # HxWx3
    h_coords = torch.stack([x_grid_coords, y_grid_coords, torch.ones_like(x_grid_coords)], -1)
    rays = h_coords @ K.inverse().T
    ray_scale = rays.norm(p=2, dim=-1)

    return NF.normalize(rays, p=2, dim=-1), ray_scale

def load_sample(ray, ray_scale, height, width, sample):
    '''
    sample = {
        'image_file': Path,
        'pose': 4x4 opencv convention transform matrix.
    }
    '''
    image_file = sample['image_file']
    pose = sample['pose']

    with Image.open(image_file) as img:
        rgba_ = F.resize(F.to_tensor(img), (height, width))
        if rgba_.shape[0] == 4:
            alpha = rgba_[3:]
            rgb = rgba_[:3]
        elif rgba_.shape[0] == 3:
            rgb = rgba_[:3]
            alpha = torch.ones_like(rgb[:1])
        rgb = rgb.view(3, -1).T
        alpha = alpha.view(1, -1).T
        
    # orient ray
    ray_d = ray @ pose[:3, :3].T
    ray_o = pose[:3, 3].expand(ray_d.shape)

    data = {
      'ray_o': ray_o.view(-1, 3), 
      'ray_d': ray_d.view(-1, 3), 
      'ray_od': ray.view(-1, 3), 
      'ray_s': ray_scale.view(-1, 1),
      'rgb': rgb.view(-1, 3), 
      'a': alpha.view(-1, 1),
    }
    
    if 'normal_file' in sample:
        normal_file = sample['normal_file']
        with Image.open(normal_file) as img:
            normal = F.resize(F.to_tensor(img), (height, width))
        normal = normal * 2 - 1
        normal = NF.normalize(normal, p=2, dim=0)
        data['n'] = normal.permute(1, 2, 0).view(-1, 3)

    if 'depth_file' in sample:
        depth_file = sample['depth_file']
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        depth = F.resize(depth, (height, width))
        data['d'] = depth.view(-1, 1)

    if 'mask_file' in sample:
        mask_file = sample['mask_file']
        with Image.open(mask_file) as img:
            mask = 1 - F.resize(F.to_tensor(img), (height, width))

        # 1xHxW
        data['m'] = mask[0].view(-1, 1)

    return data


def load_sample_eval(ray, ray_scale, height, width, sample):
    '''
    sample = {
        'image_file': Path,
        'pose': 4x4 opencv convention transform matrix.
    }
    '''
    pose = sample['pose']
    # iid = sample['iid']

    alpha = torch.ones_like(ray_scale)
        
    # orient ray
    ray_d = ray @ pose[:3, :3].T
    ray_o = pose[:3, 3].expand(ray_d.shape)

    data = {
    #   'iid': iid,
      'ray_o': ray_o.view(-1, 3), 
      'ray_d': ray_d.view(-1, 3), 
      'ray_od': ray.view(-1, 3), 
      'ray_s': ray_scale.view(-1, 1),
      'a': alpha.view(-1, 1),
    }

    return data
