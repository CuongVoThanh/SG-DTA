import os
import re
import logging

import torch
from torch.utils.tensorboard import SummaryWriter


class Experiment:
    def __init__(self, 
                exp_name: str,
                model_checkpoint_interval: int = 100, 
                mode: str = 'train', 
                exps_basedir: str ='experiments',
                tensorboard_dir: str = 'tensorboard') -> None:
        self.name = exp_name
        self.exp_dirpath = os.path.join(exps_basedir, exp_name)
        self.models_dirpath = os.path.join(self.exp_dirpath, 'models')
        self.results_dirpath = os.path.join(self.exp_dirpath, 'results')
        self.log_path = os.path.join(self.exp_dirpath, f'log_{mode}.txt')
        self.tensorboard_writer = SummaryWriter(os.path.join(tensorboard_dir, exp_name))
        self.model_checkpoint_interval = model_checkpoint_interval
        self.setup_exp_dir()
        self.setup_logging()

    def setup_exp_dir(self) -> None:
        if not os.path.exists(self.exp_dirpath):
            os.makedirs(self.exp_dirpath)
            os.makedirs(self.models_dirpath)
            os.makedirs(self.results_dirpath)

    def setup_logging(self) -> None:
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])
        self.logger = logging.getLogger(__name__)

    def get_last_checkpoint_epoch(self) -> int:
        pattern = re.compile('fold_(\\d+)_model_(\\d+).pt')
        last_epoch = -1
        for ckpt_file in os.listdir(self.models_dirpath):
            result = pattern.match(ckpt_file)
            if result is not None:
                epoch = int(result.groups()[1])
                if epoch > last_epoch:
                    last_epoch = epoch
        return last_epoch

    def get_checkpoint_path(self, epoch: int, fold: int):
        return os.path.join(self.models_dirpath, 'fold_{:04d}_model_{:04d}.pt'.format(fold, epoch))

    def get_epoch_model(self, epoch: int, fold: int):
        return torch.load(self.get_checkpoint_path(epoch, fold))['model']

    def load_last_train_state(self,
                            model: torch.nn.Module,
                            optimizer: torch.optim,
                            scheduler: torch.optim.lr_scheduler,
                            best_mse: float,
                            fold: int) -> tuple:
        epoch = self.get_last_checkpoint_epoch()
        train_state_path = self.get_checkpoint_path(epoch, fold)
        train_state = torch.load(train_state_path)
        model.load_state_dict(train_state['model'])
        optimizer.load_state_dict(train_state['optimizer'])
        scheduler.load_state_dict(train_state['scheduler'])
        if train_state['best_mse']:
            best_mse = train_state['best_mse']
        return epoch, model, optimizer, scheduler, best_mse

    def save_train_state(self, 
                        epoch: int,
                        model: torch.nn.Module,
                        optimizer: torch.optim, 
                        scheduler: torch.optim.lr_scheduler,
                        fold: int,
                        best_mse: float) -> None:
        train_state_path = self.get_checkpoint_path(epoch, fold)
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mse': best_mse,
            }, train_state_path)

    def iter_end_callback(self, 
                        epoch: int, 
                        max_epochs: int,
                        iter_nb: int, 
                        max_iter: int,
                        loss: float,
                        lr: float,
                        fold: int) -> None:
        line = 'Epoch [{}/{}] - Iter [{}/{}] - Loss: {:.5f} - Lr: {:.5f}'.format(epoch, max_epochs, iter_nb + 1, max_iter, loss, lr)
        self.logger.debug(line)
        overall_iter = (epoch * max_iter) + iter_nb
        self.tensorboard_writer.add_scalar(f'loss/total_loss on fold {fold}', loss, overall_iter)
        self.tensorboard_writer.add_scalar(f'lr/fold {fold}', lr, overall_iter)

    def epoch_start_callback(self, epoch: int, max_epochs: int) -> None:
        self.logger.debug('Epoch [%d/%d] starting.', epoch, max_epochs)

    def epoch_end_callback(self,
                        epoch: int, 
                        max_epochs: int, 
                        model: torch.nn.Module, 
                        optimizer: torch.optim,
                        scheduler: torch.optim.lr_scheduler,
                        fold: int,
                        best_mse: float) -> None:
        self.logger.debug('Epoch [%d/%d] finished.\n', epoch, max_epochs)

        # if epoch % self.model_checkpoint_interval == 0:
        self.save_train_state(epoch, model, optimizer, scheduler, fold, best_mse)

    def delete_model(self, 
                    is_not_delete_model: bool,
                    fold: int,
                    epoch: int) -> None:
        current_model = 'fold_{:04d}_model_{:04d}.pt'.format(fold, epoch)

        if len(os.listdir(self.models_dirpath)):
            for content in os.listdir(self.models_dirpath):
                if (content != current_model and is_not_delete_model) or (content == current_model and not is_not_delete_model):
                    os.remove(os.path.join(self.models_dirpath, content))
    
    def train_start_callback(self) -> None:
        self.logger.debug('Beginning training session.\n')

    def train_end_callback(self) -> None:
        self.logger.debug('Training session finished.\n')

    def eval_start_callback(self) -> None:
        self.logger.debug('Beginning testing session.\n')

    def eval_end_callback(self,
                        mse_score: float,
                        ci_score: float,
                        spearman_score: float, 
                        pearson_score: float,
                        epoch_evaluated: int,
                        on_val: bool,
                        fold: int) -> None:
        mode = 'validation' if on_val else 'test'
        
        # log tensorboard metrics
        self.tensorboard_writer.add_scalar(f'MSE_{mode}/fold {fold}', mse_score, epoch_evaluated)
        self.tensorboard_writer.add_scalar(f'CI_{mode}/fold {fold}', ci_score, epoch_evaluated)
        self.tensorboard_writer.add_scalar(f'spearman_{mode}/fold {fold}', spearman_score, epoch_evaluated)
        self.tensorboard_writer.add_scalar(f'pearson_{mode}/fold {fold}', pearson_score, epoch_evaluated)
        
        if on_val:
            self.logger.debug(f'{mode} session finished on model after epoch {epoch_evaluated}.\n')
        else:
            self.logger.debug(f'{mode} session finished.\n')

        self.logger.info(f"MSE {mode}: {mse_score:.5f}, CI {mode}:{ci_score:.5f}, spearman {mode}:{spearman_score:.5f}, pearson {mode}:{pearson_score:.5f}.\n")