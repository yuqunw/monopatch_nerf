import sys
from pathlib import Path
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import time
import math
import torch.optim as opt
import torch.nn.functional as NF

from .visualization import *
from .checkpoints import Checkpointer
from .logger import Logger
from .loss import compute_loss, compute_generate_loss
from .image import *
from .arg_dict_class import ArgDictClass
from src.data.patch_datasets import PatchCachedDataset, PatchSampleDataset

class Trainer(ArgDictClass):
    """Trainer utility class.

    Trainer class should handle 3 things:
        1) saving / loading of checkpoints at each val steps
        2) running a validation step
        3) logging the training / validation details
    """
    def __init__(self, model, grid,
                      experiment_name: str,
                      experiment_path:Path,
                      device='cuda',

                      # training options
                      lr = 1e-2,
                      max_iters=200000,

                      # loss
                      dist_loss_weight = 0.0,
                      entropy_loss_weight = 0.01,
                      normal_loss_weight= 0.0005,
                      density_normal_loss_weight= 0.0005,
                      density_normal_grad_loss_weight= 0.0005,
                      depth_loss_weight= 0.05,
                      depth_grad_loss_weight= 0.05,
                      ncc_loss_weight= 0.005,
                      ssim_loss_weight= 0.0005, 
                      normal_smooth_weight= 0.0005,
                      generate_patch = True,

                      max_smooth_steps = 200000,
                      max_monocular_smooth_steps = 200000,

                      # validation 
                      val_freq=5000,
                      fp_16=False,
                      resume=False,
        ):
        super().__init__()
        """
        On init, the trainer should do:
            1) setup paths based on the arguments
            2) load checkpoint if the training should resume
        """
        # setup sub classes
        experiment_path = Path(experiment_path)
        self.logger = Logger(experiment_path, experiment_name)
        self.checkpointer = Checkpointer(experiment_path)
        self.val_freq = val_freq
        self.lr = lr
        self.optimizer = opt.Adam(model.parameters(), lr=self.lr, eps=1e-15)
        self.max_iters =  max_iters
        self.max_monocular_loss_steps = max_monocular_smooth_steps

        self.scheduler = opt.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[max_iters // 2, max_iters * 3 // 4, max_iters * 9 // 10],
            gamma=0.33,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=fp_16)

        self.fp_16 = fp_16
        self.loss_weights = {
            'rgb': 1.0,
            'mask': 0.01,
            'ent': entropy_loss_weight, 
            'dist': dist_loss_weight, 
            'ent': entropy_loss_weight,
            'n': normal_loss_weight,
            'gn': density_normal_loss_weight,
            'gns': density_normal_grad_loss_weight,
            'd': depth_loss_weight,
            'dg': depth_grad_loss_weight,
            'ncc': ncc_loss_weight,
            'ssim': ssim_loss_weight,
            'nsl': normal_smooth_weight            
        }
        self.val_batch = 1024

        # setup counter variables
        self.global_step = 0

        self.device = torch.device(device)
        self.generate_patch = generate_patch
        self.max_smooth_steps = max_smooth_steps
        self.max_monocular_smooth_steps = max_monocular_smooth_steps


        if resume:
            self.checkpointer.load_model_checkpoint(model, device=device)
            self.checkpointer.load_grid_checkpoint(grid, device=device)
            self.checkpointer.load_train_checkpoint(self, device=device)

    @property
    def monocular_loss_weights(self):
        return math.exp(-self.global_step / self.max_monocular_loss_steps * 10.0)

    def fit(self, model, grid, datamodule):
        # iterate steps for training
        train_dataset = datamodule.train_dataset
        model.train()
        grid.train()

        iterator = tqdm(enumerate(train_dataset), total=self.max_iters)
        # iterator = enumerate(train_dataset)

        print(f'Training starts!', flush=True)
        start_time = time.time()
        for _, batch in iterator:
            # run validation every few steps
            if ((self.global_step % self.val_freq) == 0) and (self.global_step != 0):
            # if ((self.global_step % self.val_freq) == 0):
                model.eval()
                grid.eval()
                self.validate(model, grid, datamodule)

                # save checkpoint
                self.checkpointer.save_checkpoint(self, model, grid, self.global_step)
                model.train()
                grid.train()
            if batch['a'].sum() == 0:
                print('no training data found!', flush=True)
                continue
                
            with torch.cuda.amp.autocast(enabled=self.fp_16):
                grid.update(self.global_step)

            res = self.train_step(model, grid, batch, iterator, self.global_step)

            # update patches
            if res is not None:
                self.global_step += 1

                num_rays = len(batch['ray_o'])
                n_rendering_samples = res['n']
                train_dataset.samples_per_batch = n_rendering_samples / num_rays

            if (self.global_step % 1000) == 0:
                curr_time = time.time()
                time_taken = curr_time - start_time
                time_per_step = time_taken / (self.global_step)
                time_left = time_per_step * (self.max_iters - self.global_step)
                minutes_left = int(time_left // 60)
                hour = int(minutes_left // 60)
                second = int(time_left % 60)
                minute = int(minutes_left % 60)
                print(f'Trained step {self.global_step} / {self.max_iters} (eta {hour:02d}:{minute:02d}:{second:02d})', flush=True)

            if self.global_step > self.max_iters:
                break

    def validate(self, model, grid, datamodule):
        model.eval()
        W = datamodule.val_width
        H = datamodule.val_height
        val_dataset = datamodule.val_dataset

        psnrs = []
        for i, batch in enumerate(val_dataset):

            # render inf_images
            gts = render_gt_image(batch, H, W)
            infs = render_image(batch, model, grid, (H, W), self.val_batch, self.fp_16)

            mse = NF.mse_loss(infs['rgb'], gts['rgb'])
            psnr = -10.0 * math.log10(mse.item())
            psnrs.append(psnr)
            if i == 0:
                renderable = combine_renderables(gts, infs)


        logs = {}
        file_logs = {}
        logs['val/psnr'] = np.array(psnr).mean()
        logs[f'val/rendered'] = renderable
        self.logger.log(logs, self.global_step)

        file_logs[f'val/rendered'] = F.to_pil_image(renderable)
        self.logger.log_to_files(file_logs, self.global_step)
        torch.cuda.empty_cache()


    def train_step(self, model, grid, batch, iterator, global_step):
        self.optimizer.zero_grad()

        # compute loss
        with torch.cuda.amp.autocast(enabled=self.fp_16):
            ray_o = batch['ray_o']
            ray_d = batch['ray_d']
            B = ray_o.shape[0]

            ray_o = ray_o.view(-1, 3)
            ray_d = ray_d.view(-1, 3)
            ray_i, t_starts, t_ends = grid.sample(ray_o, ray_d)
            n_rendering_samples = len(t_starts)
            if n_rendering_samples  == 0:
                print('rendering nothing.. returning')
                return None
            res = model.render(ray_o, ray_d, ray_i, t_starts, t_ends)
            losses = compute_loss(res, batch, self.loss_weights, self.monocular_loss_weights)

        if self.generate_patch and 'rand_o' in batch:
            # compute random ray direction
            with torch.cuda.amp.autocast(enabled=self.fp_16):
                ray_o = batch['ray_o'].view(-1, 3)
                ray_d = batch['ray_d'].view(-1, 3)
                novel_o = batch['rand_o'].view(-1, 3)
                patch_world_coor = ray_o + ray_d * res['d'].view(-1, 1)
                novel_d = NF.normalize(patch_world_coor - novel_o, dim=-1, p=2)
                batch['rand_d'] = novel_d
                novel_i, nt_starts, nt_ends = grid.sample(novel_o, novel_d)
                rand_n_rendering_samples = len(nt_starts)
                if rand_n_rendering_samples > 0:
                    novel_res = model.render(novel_o, novel_d, novel_i, nt_starts, nt_ends)
                    gen_losses = compute_generate_loss(batch, novel_res)
                    losses['loss'] += (gen_losses['ncc_loss'] * self.loss_weights.get('ncc', 0.001)) + \
                                      (gen_losses['ssim_loss'] * self.loss_weights.get('ssim', 0.0005)) + \
                                      (gen_losses['n_smooth_loss'] * self.loss_weights.get('nsl', 0.0005))
                    losses = {
                        **gen_losses,
                        **losses
                    }
        if losses['loss'].isnan():
            print('Nan loss found',  file=sys.stderr)
            
        self.grad_scaler.scale(losses['loss']).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.scheduler.step()
        lr_factor = self.scheduler.get_last_lr()[0] / self.lr


        # log results
        logs = {
            f'train/{k}': v
            for k, v in losses.items()
        }
        logs['train/lr_factor'] = lr_factor

        cli_log = f'loss: {losses["loss"]:.3f}|psnr: {losses["psnr"]:.2f}| lrf: {lr_factor:.04f}'
        loggables = {'n_s': n_rendering_samples, 'batch': B}
        for k, v in loggables.items():
            cli_log += f' | {k}: {v}'
            logs[f'train/{k}'] = v

        self.logger.log(logs, self.global_step)
        # iterator.set_description(cli_log)

        return { 'n': n_rendering_samples }
