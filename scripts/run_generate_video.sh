python scripts/eval_video.py --model_checkpoint_file ~/facade/checkpoints/model/model_steps_50000.ckpt \
                      --grid_checkpoint_file ~/facade/checkpoints/grid/grid_steps_50000.ckpt \
                      --output_path /mnt/data/eth3d_video_saved/facade \
                      --full True

# python scripts/train.py --data_path /mnt/data/eth3d_processed/kicker \
#                         --output_path outputs \
#                         --experiment_name kicker \
#                         --patch_sampling True \
#                         --depth_loss_weight 0.05 \
#                         --depth_grad_loss_weight 0.025 \
#                         --normal_loss_weight 0.001 \
#                         --density_normal_loss_weight 0.0005 \
#                         --density_normal_grad_loss_weight 0.0001 \
#                         --normal_smooth_weight 0.0005 \
#                         --generate_patch True \
#                         --ssim_loss_weight 0.0005 \
#                         --ncc_loss_weight 0.001 \
#                         --full True \
#                         --cull True \
#                         --max_iters 50000 \
#                         --val_freq 50000      

# python scripts/eval_video.py --model_checkpoint_file outputs/kicker/checkpoints/model/model_steps_50000.ckpt \
#                       --grid_checkpoint_file outputs/kicker/checkpoints/grid/grid_steps_50000.ckpt \
#                       --output_path /mnt/data/eth3d_video/kicker \
#                       --full True

# python scripts/train.py --data_path /mnt/data/eth3d_processed/relief_2 \
#                         --output_path outputs \
#                         --experiment_name relief_2 \
#                         --patch_sampling True \
#                         --depth_loss_weight 0.05 \
#                         --depth_grad_loss_weight 0.025 \
#                         --normal_loss_weight 0.001 \
#                         --density_normal_loss_weight 0.0005 \
#                         --density_normal_grad_loss_weight 0.0001 \
#                         --normal_smooth_weight 0.0005 \
#                         --generate_patch True \
#                         --ssim_loss_weight 0.0005 \
#                         --ncc_loss_weight 0.001 \
#                         --full True \
#                         --cull True \
#                         --max_iters 50000 \
#                         --val_freq 50000      

# python scripts/eval_video.py --model_checkpoint_file outputs/relief_2/checkpoints/model/model_steps_50000.ckpt \
#                       --grid_checkpoint_file outputs/relief_2/checkpoints/grid/grid_steps_50000.ckpt \
#                       --output_path /mnt/data/eth3d_video/relief_2 \
#                       --full True

# python scripts/train.py --data_path /mnt/data/eth3d_processed/terrace \
#                         --output_path outputs \
#                         --experiment_name terrace \
#                         --patch_sampling True \
#                         --depth_loss_weight 0.05 \
#                         --depth_grad_loss_weight 0.025 \
#                         --normal_loss_weight 0.001 \
#                         --density_normal_loss_weight 0.0005 \
#                         --density_normal_grad_loss_weight 0.0001 \
#                         --normal_smooth_weight 0.0005 \
#                         --generate_patch True \
#                         --ssim_loss_weight 0.0005 \
#                         --ncc_loss_weight 0.001 \
#                         --full True \
#                         --cull True \
#                         --max_iters 50000 \
#                         --val_freq 50000      

# python scripts/eval_video.py --model_checkpoint_file outputs/terrace/checkpoints/model/model_steps_50000.ckpt \
#                       --grid_checkpoint_file outputs/terrace/checkpoints/grid/grid_steps_50000.ckpt \
#                       --output_path /mnt/data/eth3d_video/terrace \
#                       --full True                      

# python scripts/train.py --data_path /mnt/data/eth3d_processed/courtyard \
#                         --output_path outputs \
#                         --experiment_name courtyard \
#                         --patch_sampling True \
#                         --depth_loss_weight 0.05 \
#                         --depth_grad_loss_weight 0.025 \
#                         --normal_loss_weight 0.001 \
#                         --density_normal_loss_weight 0.0005 \
#                         --density_normal_grad_loss_weight 0.0001 \
#                         --normal_smooth_weight 0.0005 \
#                         --generate_patch True \
#                         --ssim_loss_weight 0.0005 \
#                         --ncc_loss_weight 0.001 \
#                         --full True \
#                         --cull True \
#                         --max_iters 50000 \
#                         --val_freq 50000      

# python scripts/eval_video.py --model_checkpoint_file outputs/courtyard/checkpoints/model/model_steps_50000.ckpt \
#                       --grid_checkpoint_file outputs/courtyard/checkpoints/grid/grid_steps_50000.ckpt \
#                       --output_path /mnt/data/eth3d_video/courtyard \
#                       --full True                      

