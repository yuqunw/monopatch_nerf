import sys
sys.path.append('.')
sys.path.append('..')

import torch
from pathlib import Path
from datetime import datetime
from src.data.datamodule import DataModule
from src.model.neural_model import NeuralModel
from src.model.nerfacc_sampler import NeRFAccSampler
from src.utils.trainer import Trainer
import math
import os

def main(args):
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    datamodule = DataModule(args.data_path, 
                            args.batch_size, 
                            args.target_batch_size,
                            args.num_workers, 
                            args.width, args.height, args.val_inv_scale,
                            patch_sampling=args.patch_sampling,
                            device=args.device,
                            full=args.full,
                            use_mask = args.use_mask)

    model = NeuralModel(
        num_feats=args.num_feats,
        min_log2_freq=args.min_log2_freq,
        max_log2_freq=args.max_log2_freq,
        num_freqs=args.num_freqs,
        num_quants=args.num_quants,
        use_density_normal=args.use_density_normal,
        use_normal=args.use_normal,
        use_seg=args.use_seg
    )
    model.to(args.device)
    grid = NeRFAccSampler(density_query=model.query_density, target_step_size=2048)
    grid = grid.to(args.device)
    grid.init()
    grid.poses = datamodule.poses
    grid.camera = datamodule.camera
    if args.cull:
        grid.cull(datamodule, culling_thre = args.culling_thre)

    # train
    output_path = Path(args.output_path)
    exp_path =  output_path / args.experiment_name
    trainer = Trainer(model, grid,
                      experiment_name=args.experiment_name, 
                      experiment_path=exp_path, 
                      lr=args.lr,
                      max_iters=args.max_iters,
                      val_freq=args.val_freq,
                      dist_loss_weight = args.dist_loss_weight,
                      entropy_loss_weight = args.entropy_loss_weight,
                      normal_loss_weight= args.normal_loss_weight,
                      density_normal_loss_weight= args.density_normal_loss_weight,
                      density_normal_grad_loss_weight= args.density_normal_grad_loss_weight,
                      depth_loss_weight= args.depth_loss_weight,
                      depth_grad_loss_weight= args.depth_grad_loss_weight,
                      ncc_loss_weight = args.ncc_loss_weight,
                      ssim_loss_weight = args.ssim_loss_weight,
                      normal_smooth_weight = args.normal_smooth_weight, 
                      generate_patch = args.generate_patch,
                      device=args.device)
    trainer.logger.log_args(args)
    trainer.logger.log_model(model)
    trainer.logger.log_git()
    trainer.fit(model, grid, datamodule)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # set data / training arguments
    parser.add_argument('--data_path', help="Path to dataset")
    parser.add_argument('--output_path', default='outputs', help="Path to logging, checkpointing outputs")
    parser.add_argument('--experiment_name', default='debug')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint_path', default=None, help="Use to resume training")

    # model specific args
    parser.add_argument('--num_feats', default=4, type=int)
    parser.add_argument('--min_log2_freq', default=0, type=int)
    parser.add_argument('--max_log2_freq', default=5, type=int)
    parser.add_argument('--num_freqs', default=4, type=int)
    parser.add_argument('--num_quants', default=80, type=int)
    parser.add_argument('--use_normal', default='True', type=eval, choices=[True, False])
    parser.add_argument('--use_density_normal', default='True', type=eval, choices=[True, False])
    parser.add_argument('--use_seg', default='False', type=eval, choices=[True, False])
    parser.add_argument('--culling_thre', default=0, type=int, help='Density restriction thresholding')

    # trainer specific args
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--max_iters', default=50000, type=int)
    parser.add_argument('--val_freq', type=int, default=10000)
    parser.add_argument('--dist_loss_weight', type=float, default=0.0)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)
    parser.add_argument('--depth_loss_weight', type=float, default=0.05)
    parser.add_argument('--depth_grad_loss_weight', type=float, default=0.025)
    parser.add_argument('--normal_loss_weight', type=float, default=0.001)
    parser.add_argument('--density_normal_loss_weight', type=float, default=0.0005)
    parser.add_argument('--density_normal_grad_loss_weight', type=float, default=0.0001)
    parser.add_argument('--ncc_loss_weight', type=float, default=0.001)
    parser.add_argument('--ssim_loss_weight', type=float, default=0.0005)
    parser.add_argument('--normal_smooth_weight', type=float, default=0.0005)
    parser.add_argument('--generate_patch', default='True', type = eval, choices=[True, False], 
                        help='Whether generate virtual patches')

    # data specific args
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--target_batch_size', default=2**20, type=int)
    parser.add_argument('--patch_sampling', default='True', type=eval, choices=[True, False], help='Sample patches or rays during rendering')
    parser.add_argument('--width', default=None, type=int)
    parser.add_argument('--height', default=None, type=int)
    parser.add_argument('--val_inv_scale', type=int, default=1)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--full', default='True', type=eval, choices=[True, False], help='Use all input views or 9/10 views for training')
    parser.add_argument('--cull', default='True', type=eval, choices=[True, False], help='Whether using density restriction')
    parser.add_argument('--use_mask', default='True', type=eval, choices=[True, False], help='Whether using mask during loading images')

    args = parser.parse_args()
    main(args)
