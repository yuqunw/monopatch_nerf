import torch
import logging
from pathlib import Path

class Checkpointer:
    def __init__(self, experiment_path):
        self.exp_path = Path(experiment_path)
        self.checkpoint_path = self.exp_path / 'checkpoints'
        self.model_path = self.checkpoint_path / 'model'
        self.grid_path = self.checkpoint_path / 'grid'
        self.train_path = self.checkpoint_path / 'train'

        self.model_path.mkdir(parents=True, exist_ok=True)
        self.grid_path.mkdir(parents=True, exist_ok=True)
        self.train_path.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, trainer, model, grid, step=None):
        """
        """
        checkpoint_filename = 'last.ckpt'
        if(step!= None):
            checkpoint_filename = f'steps_{step:d}.ckpt'

        model_checkpoint_path = self.model_path / ('model_' + checkpoint_filename)
        grid_checkpoint_path = self.grid_path / ('grid_' + checkpoint_filename)
        train_checkpoint_path = self.train_path / ('trainer_' + checkpoint_filename)

        model_ckpt = {
            'state_dict': model.state_dict(),
            'class': type(model),
            'arg_dict': model.args_dict
        }
        grid_ckpt = {
            'state_dict': grid.state_dict(),
            'class': type(grid),
            'arg_dict': grid.args_dict,
            'poses': grid.poses
        }

        train_ckpt = {
            'global_step': trainer.global_step,
            'optimizer': trainer.optimizer.state_dict(),
            'scheduler': trainer.scheduler.state_dict(),
            'args': trainer.args_dict,
            'class': type(trainer)
        }

        torch.save(model_ckpt, model_checkpoint_path)
        torch.save(grid_ckpt, grid_checkpoint_path)
        torch.save(train_ckpt, train_checkpoint_path)
        

    def load_grid_checkpoint(self, grid=None, grid_checkpoint_file=None, device='cpu'):
        """loads checkpoint by restoring trainer state

        :param checkpoint_path: full path to the checkpoint file. If None, trainer will check based on 
                                experiment name and resume from the last checkpoint
        """
        if(grid_checkpoint_file is None):
            grid_checkpoint_file = self.model_path / 'grid_last.ckpt'
            if not (grid_checkpoint_file.exists()):
                logging.info("No valid checkpoint found. Starting training from scratch!")
                return
        else:
            raise Exception('No Checkpoint Found')
        
        grid_ckpt = torch.load(grid_checkpoint_file, device)
        if grid is None:
            grid = grid_ckpt['class'](**grid_ckpt['args'])

        grid.load_state_dict(grid_ckpt['state_dict'])
        return grid 

    def load_model_checkpoint(self, model=None, model_checkpoint_file=None, device='cpu'):
        """loads checkpoint by restoring trainer state

        :param checkpoint_path: full path to the checkpoint file. If None, trainer will check based on 
                                experiment name and resume from the last checkpoint
        """
        if(model_checkpoint_file is None):
            model_checkpoint_file = self.model_path / 'model_last.ckpt'
            if not (model_checkpoint_file.exists()):
                logging.info("No valid checkpoint found. Starting training from scratch!")
                return
        else:
            raise Exception('No Checkpoint Found')
        
        model_ckpt = torch.load(model_checkpoint_file, device)
        if model is None:
            model = model_ckpt['class'](**model_ckpt['args'])

        model.load_state_dict(model_ckpt['state_dict'])
        return model

    def load_train_checkpoint(self, trainer=None, train_checkpoint_file=None, device='cpu'):
        if(train_checkpoint_file is None):
            train_checkpoint_file = self.train_path / 'trainer_last.ckpt'
            if not (train_checkpoint_file.exists()):
                logging.info("No valid checkpoint found. Starting training from scratch!")
                return
        else:
            raise Exception('No Checkpoint Found')

        train_ckpt = torch.load(train_checkpoint_file, device)
        if trainer is None:
            trainer = train_ckpt['class'](**train_ckpt['args'])

        trainer.optimizer.load_state_dict(train_ckpt['optimizer'])
        trainer.scheduler.load_state_dict(train_ckpt['scheduler'])
        trainer.global_step = train_ckpt['global_step']