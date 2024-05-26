from pathlib import Path
from PIL.Image import Image
import torch
from torchvision.transforms.functional import to_tensor
import pandas as pd
import trimesh

from tensorboardX import SummaryWriter
class Logger():
    def __init__(self, experiment_path, experiment_name):
        self.root = Path(experiment_path)
        self.log_path = self.root / 'logs'
        self.val_path = self.root / 'vals'
        self.log_path.mkdir(exist_ok=True, parents=True)
        self.val_path.mkdir(exist_ok=True, parents=True)

        self.logger = SummaryWriter(self.log_path)
        self.experiment_name = experiment_name

    def log_config(self, cfg):
        detail_dict = {}
        def log_child(prefix, c):
            for k, v in c.items():
                if hasattr(v, 'items'):
                    log_child(f'{prefix}.{k}', v)
                else:
                    detail_dict[f'{prefix}.{k}'] = v
        log_child('cfg',cfg)

        # add markdown to text
        detail_df = pd.DataFrame.from_dict(detail_dict, orient='index', columns=['Configs'])
        print('-- Configs')
        print(detail_df)
        self.logger.add_text('config', detail_df.to_markdown())
    def log_args(self, args):
        detail_dict = {}
        def log_child(prefix, c):
            for k, v in c.items():
                if hasattr(v, 'items'):
                    log_child(f'{prefix}.{k}', v)
                else:
                    detail_dict[f'{prefix}.{k}'] = v
        log_child('cfg',vars(args))

        # add markdown to text
        detail_df = pd.DataFrame.from_dict(detail_dict, orient='index', columns=['Configs'])
        print('-- Configs')
        print(detail_df)
        self.logger.add_text('config', detail_df.to_markdown())
    def log_git(self)->None:
        import os
        commit_id = os.popen('git rev-parse HEAD').read()
        self.logger.add_text('code/commit_id', commit_id.strip())


    def log_model(self, model)->None:
        summary_dict = {}
        detail_dict = {}

        summary_str = ['='*80]
        num_params = 0

        for module_name, module in model.named_children():
            summary_count = 0
            for name, param in module.named_parameters():
                if(param.requires_grad):
                    detail_dict[name] = [str(tuple(param.shape))]
                    summary_count += param.numel()
                    num_params += param.numel()
            summary_dict[module_name] = [summary_count]
            summary_str+= [f'- {module_name: <40} : {str(summary_count):^34s}']

        detail_dict['total'] = [num_params]
        summary_dict['total'] = [num_params]

        # print summary string
        summary_str += ['='*80]
        summary_str += ['--' +  f'{"Total":<40} : {str(num_params) + " params":^33s}' +'--']
        summary_str += ['='*80]
        print('\n'.join(summary_str))

        # add markdown to text
        summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=['Count'])
        detail_df = pd.DataFrame.from_dict(detail_dict, orient='index', columns=['Shape'])
        # print('-- Detail')
        # print(detail_df)
        self.logger.add_text('model/summary', summary_df.to_markdown())
        self.logger.add_text('model/detail', detail_df.to_markdown())

    def log_to_files(self, loggables, log_step):
        for k, v in loggables.items():
            out_prefix = self.val_path / f'{self.experiment_name}_step_{log_step}_{k.replace("/", "-")}'
            if isinstance(v, tuple) and len(v) == 3:
                (verts, faces, colors) = v
                if verts is not None:
                    mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors)
                    mesh.export(str(out_prefix) + '.ply')
            elif isinstance(v, tuple) and len(v) == 2:
                (verts, colors) = v
                if verts is not None:
                    mesh = trimesh.points.PointCloud(verts, colors)
                    mesh.export(str(out_prefix) + '.ply')
            elif isinstance(v, Image):
                v.save(str(out_prefix) + '.png')
                self.logger.add_image(k, to_tensor(v), log_step)

    def log(self, loggables: dict, log_step: int)->None:
        for k, v in loggables.items():
            if isinstance(v, Image):
                self.logger.add_image(k, to_tensor(v), log_step)
            elif isinstance(v, torch.Tensor):
                if len(v.shape) == 5:
                    assert v.shape[2] == 3
                    self.logger.add_video(k, v, global_step=log_step, fps=2)
                elif len(v.shape) == 4:
                    dataformats = 'NCHW'
                    if v.shape[-1] == 3:
                        dataformats = 'NHWC'
                    self.logger.add_images(k, v, global_step=log_step, dataformats=dataformats)
                elif len(v.shape) == 3:
                    if v.shape[0] == 3:
                        dataformat = 'CHW'
                    elif v.shape[-1] == 3:
                        dataformat = 'HWC'
                    self.logger.add_image(k, v, log_step, dataformats=dataformat)
                else:
                    self.logger.add_scalar(k, v, log_step)
            else:
                self.logger.add_scalar(k, v, log_step)
        self.logger.flush()
    