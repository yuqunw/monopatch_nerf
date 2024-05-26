import math
import torch.nn.functional as NF
import torch
from .masked_ssim import ssim

def compute_ncc_loss(img_patch, generated_patch, mask):
    # B, H, W, 3
    valid_img_patch = img_patch * mask
    norm_img_patch = valid_img_patch - valid_img_patch.sum(dim=(1, 2), keepdim=True) / (mask.sum(dim=(1, 2), keepdim=True) + 1e-8)
    norm_img_patch = NF.normalize(norm_img_patch, p=2, dim=(1, 2))

    valid_generate_patch = generated_patch * mask
    norm_generate_patch = valid_generate_patch - valid_generate_patch.sum(dim=(1, 2), keepdim=True) / (mask.sum(dim=(1, 2), keepdim=True) + 1e-8)
    norm_generate_patch = NF.normalize(norm_generate_patch, p=2, dim=(1, 2),)

    # ncc = (norm_img_patch * norm_generate_patch).sum() / img_patch.shape[0] / 3
    ncc = (norm_img_patch * norm_generate_patch).sum(dim = (1, 2, 3))
    ncc = (ncc * mask.sum(dim=(1, 2, 3))).sum() / (mask.sum() + 1e-8) # Weight avergae for each batch
    # if torch.isnan(ncc):
    #     print('nan loss, stop here! You shall not passsssssss!!!')    

    # Update NCC loss
    ncc = ncc / 3 # 3 channels
    ncc_loss = ((1-ncc)**2) # -1 ~ 1 => 4 ~ 0
    return ncc_loss


def compute_ssim_loss(img_patch, generated_patch, mask):
    win_size = 5
    cut_mask = mask[:, win_size // 2 : - (win_size // 2), win_size // 2 : - (win_size // 2)]
    ssim_loss = 1 - ssim(img_patch.permute(0, 3, 1, 2), generated_patch.permute(0, 3, 1, 2), data_range=1, size_average=False, win_size = win_size) # B, H, W
    ssim_loss = (ssim_loss * cut_mask).sum() / (cut_mask.sum() + 1e-8)
    return ssim_loss

def compute_generate_loss(batch, novel_res, use_angle_for_occlusion = True, angle_threshold = 10): 
    B, PS = batch['rgb'].shape[:2]
    gt_rgb = batch['rgb'].view(B, PS, PS, 3) # 1, 3, H, W
    ref_m = batch['m'].view(B, PS, PS, 1) if ('m' in batch) else torch.ones_like(gt_rgb[..., :1]).bool()

    # Set occlusion
    angle_pass_rate = 1
    seg_pass_rate = 1

    # Get the rendered novel patch rgb and depth
    novel_rgb = novel_res['rgb'].view(B, PS, PS, -1) # B, PS, PS, 3
    mask = ref_m[..., 0].bool() # B, PS, PS, 1
    if use_angle_for_occlusion:
        ray_o = batch['ray_o'].view(B, PS, PS, -1) # B, PS, PS, 1
        ray_d = batch['ray_d'].view(B, PS, PS, -1) # B, PS, PS, 1
        novel_o = batch['rand_o'].view(B, PS, PS, -1) # B, PS, PS, 1
        novel_d = batch['rand_d'].view(B, PS, PS, -1) # B, PS, PS, 1
        inf_d = novel_res['d'].view(B, PS, PS, -1) # B, PS, PS, 1
        
        novel_patch_world_coor = novel_o + novel_d * inf_d # B, PS, PS, 3 (x, y, z)
        novel_d_wrt_refer = novel_patch_world_coor - ray_o # B, PS, PS, 3
        novel_d_wrt_refer = NF.normalize(novel_d_wrt_refer, dim = -1) # B, PS, PS, 3

        # Get the different angle between the novel patch and the reference patch in 180 degree
        diff_angle = torch.acos(torch.clamp(torch.sum(novel_d_wrt_refer * ray_d, dim = -1), -1, 1)) * 180 / torch.pi # B, PS, PS
        mask = ((diff_angle < angle_threshold) & mask).float() # B, PS, PS 
        angle_pass_rate = (diff_angle < angle_threshold).float().mean()

    ncc_loss = compute_ncc_loss(gt_rgb, novel_rgb, mask[..., None])
    ssim_loss = compute_ssim_loss(gt_rgb, novel_rgb, mask)

    # Calculate normals loss
    inf_n = NF.normalize(novel_res['n'].view(B, PS, PS, 3), p=2, dim=-1, eps=1e-8)

    normals_smooth_loss = 0 

    for scale in range(1):
        step = pow(2, scale)
        normals_smooth_loss += smoothing_loss(inf_n[:, ::step, ::step])

    return {
        'ncc_loss': ncc_loss,
        'ssim_loss': ssim_loss,
        'n_smooth_loss': normals_smooth_loss,
        'mean_mask': mask.mean(),
        'angle_pass_rate': angle_pass_rate
    }

def smoothing_loss(prediction):
    grad_x = torch.abs(1 - (prediction[:, :, 1:] * prediction[:, :, :-1]).sum(dim=-1))
    grad_y = torch.abs(1 - (prediction[:, 1:, :] * prediction[:, :-1, :]).sum(dim=-1))

    image_loss = (torch.mean(grad_x) + torch.mean(grad_y)) / 2

    return image_loss

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor
    
# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    if len(prediction.shape) == 1:
        prediction = prediction.view(1, -1, 1)
        target = target.view(1, -1, 1)
        mask = mask.view(1, -1, 1)

    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


def compute_depth_loss(inf_d, gt_d, mask_d):
    is_ray = len(inf_d.shape) == 1
    scale, shift = compute_scale_and_shift(inf_d, gt_d, mask_d)
    if is_ray:
        scale_shifted_depth = scale * inf_d + shift
    else:
        scale_shifted_depth = scale.view(-1, 1, 1) * inf_d + shift.view(-1, 1, 1)

    depth_mse = (((scale_shifted_depth - gt_d) ** 2) / (gt_d.clamp_min(0) + 0.1)) * mask_d
    depth_mse = depth_mse.sum() / mask_d.sum()
    depth_smooth_loss = 0

    if not is_ray:
        for scale in range(4):
            step = pow(2, scale)
            depth_smooth_loss += gradient_loss(scale_shifted_depth[:, ::step, ::step], gt_d[:, ::step, ::step], mask_d[:, ::step, ::step])
    return depth_mse, depth_smooth_loss

def depth_to_normal(cam_rays, depth_patch):
    B, _, PS = depth_patch.shape

    # project depth into 3d: BxPSxPSx3
    cam_points = (cam_rays * depth_patch[..., None]).permute(0, 3, 1, 2)

    sobel_filter_x = torch.tensor([
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    ]).view(3, 3)
    sobel_filter_y = sobel_filter_x.T
    # 1x2x3x3 for 3 channels
    sobel_kernel = torch.stack((sobel_filter_x, sobel_filter_y)).repeat(3, 1, 1).to(depth_patch)[:, None]

    # Bx3xPxP => Bx3x2xPxP
    grad_cam_points = NF.conv2d(cam_points, sobel_kernel, bias=None, padding=1, groups=3).view(B, 3, 2, PS, PS)

    grad_x = grad_cam_points[:, :, 0]
    grad_y = -grad_cam_points[:, :, 1]

    normal_patch = grad_x.cross(grad_y, dim=1)
    return NF.normalize(normal_patch, dim=1, p=2, eps=1e-8).permute(0, 2, 3, 1)

def compute_normal_loss(inf_n, gt_n, m):
    normal_l1_loss = NF.l1_loss(inf_n[m], gt_n[m])
    normal_ang_loss = (1 - (inf_n * gt_n).sum(-1))[m].mean()
    return normal_l1_loss + normal_ang_loss


def compute_loss(res, batch, weights={}, mono_weights=1):
    has_dist = 'dist_loss' in res
    has_depth = ('d' in res) and ('d' in batch)
    has_normal = ('n' in res) and ('n' in batch)
    has_density_normal = ('gn' in res) and ('n' in batch)
    has_gt_normal = 'n' in batch
    has_ent = 'a' in res

    gt_rgb = batch['rgb']
    gt_shape = gt_rgb.shape[:-1]
    pose = batch['pose']
    ray_s = batch['ray_s']

    gt_a = batch['a']
    rand_bg = torch.rand_like(batch['rgb'])
    is_ray = len(gt_shape) == 1

    inf_rgb = res['rgb'].view(*gt_shape, 3)
    inf_a = res['a'].view(*gt_shape, 1)
    if weights.get('ent', 0.0) == 0:
        gt_wrgb = gt_rgb * gt_a + rand_bg * (1 - gt_a)
        inf_wrgb = inf_rgb * inf_a + rand_bg * (1 - inf_a)
    else:
        gt_wrgb = gt_rgb
        inf_wrgb = inf_rgb

    m = ((batch['m'] if 'm' in batch else torch.ones_like(res['rgb'][..., :1])) > 0.5).view(*gt_shape)

    raw_losses = {
        'rgb': 0,
        'd': 0,
        'mask': 0,
        'n': 0,
        'gn': 0,
        'dist': 0,
        'ent': 0
    }
    raw_losses['rgb'] = NF.huber_loss(inf_wrgb[m], gt_wrgb[m], delta=0.1)

    
    if has_normal:
        inf_n = NF.normalize(res['n'].view(*gt_shape, 3) @ pose[:3, :3], p=2, dim=-1)
        gt_n = batch['n'].view(*gt_shape, 3)
        raw_losses['n'] = compute_normal_loss(inf_n, gt_n, m)
        if has_density_normal:
            inf_dn = NF.normalize(res['gn'].view(*gt_shape, 3) @ pose[:3, :3], p=2, dim=-1)
            grad_normal_loss = compute_normal_loss(inf_dn, gt_n, m)
            normal_smooth_loss = 0
            if not is_ray:
                for scale in range(4):
                    step = pow(2, scale)
                    normal_smooth_loss += gradient_loss(inf_dn[:, ::step, ::step], gt_n[:, ::step, ::step], m[:, ::step, ::step, None].repeat(1, 1, 1, 3))
            raw_losses['gn'] = grad_normal_loss
            raw_losses['gns'] = normal_smooth_loss
    

    if has_depth:
        inf_d = res['d'].view(*gt_shape) / ray_s.view(*gt_shape)
        gt_d = batch['d'].view(*gt_shape)
        dloss, dgloss = compute_depth_loss(inf_d, gt_d, m)
        raw_losses['d'] = dloss
        raw_losses['dg'] = dgloss

        if (not has_normal) and has_gt_normal and (not is_ray):
            gt_n = batch['n'].view(*gt_shape, 3)
            cam_ray = batch['ray_od'].view(*gt_shape, 3)
            inf_n = depth_to_normal(cam_ray, inf_d)
            raw_losses['n'] = compute_normal_loss(inf_n, gt_n, m)

    if has_ent:
        raw_losses['ent'] = (-inf_a * inf_a.clamp(1e-6, 1-1e-6).log())[m].mean()

    if has_dist:
        raw_losses['dist'] = res['dist_loss']



    loss = sum([weights[key] * raw_loss for key, raw_loss in raw_losses.items()])
    losses = {'loss': loss}
    for k, v in raw_losses.items():
        losses[f'{k}_loss'] = v

    # compute other metrics
    mse = NF.mse_loss(inf_rgb[m], gt_rgb[m])
    psnr = -10.0 * math.log10(mse.item())

    losses['psnr'] = psnr
    losses['mse'] = mse 

    return losses
