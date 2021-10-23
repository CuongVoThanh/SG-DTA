import random
import numpy as np
from tqdm import tqdm
from operator import itemgetter
import logging

import torch
from torch.utils.data import DataLoader

from utils.evaluation_metrics import mse, ci, spearman, pearson
from utils.create_data import collate

from lib.config import Config
from lib.experiment import Experiment

SEED = 42


class Runner:
    def __init__(self, 
                cfg: None or Config = None, 
                exp: None or Experiment = None, 
                device: None or torch.device = None,
                resume: int = 0,
                view: None or str = None,
                epochs: int = 1000, 
                val_on_epoch: int = 100,
                train_batch_size: int = 512,
                test_batch_size: int = 512,
                deterministic: bool = False) -> None:
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_on_epoch = val_on_epoch
        self.logger = logging.getLogger(__name__)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self, fold: int = 6) -> tuple:
        is_full_fold = True if fold == 6 else False
        starting_epoch = 1
        max_epochs = self.epochs

        self.exp.train_start_callback()
        train_loader, edge_index_drug_protein = self.get_train_dataloader(fold)

        model = self.cfg.get_model().to(self.device)

        loss_fn = self.cfg.get_loss_function()
        total_params = model.parameters()
        optimizer = self.cfg.get_optimizer(total_params)       
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        best_mse = float("inf")
        if self.resume != 0:            
            last_epoch, model, optimizer, scheduler, best_mse = self.exp.load_last_train_state(model, optimizer, scheduler, best_mse, fold)
            starting_epoch = last_epoch + 1

        for epoch in range(starting_epoch, max_epochs + 1, 1):
            model.train()
            self.exp.epoch_start_callback(epoch, max_epochs)
            pbar = tqdm(train_loader, desc=f"Train on epoch {epoch}")
            for i, data in enumerate(pbar):
                optimizer.zero_grad()
                data_drug, data_protein = map(lambda instance: instance.to(self.device), data)

                if self.cfg.config["model"]["model"] == "SGDTA":
                    output = model(data_drug, data_protein, edge_index_drug_protein)
                else:                    
                    output = model(data_drug, data_protein)

                loss = loss_fn(output, data[0].y.view(-1, 1).float().to(self.device))                    
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix(loss=loss.item())
                
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), optimizer.param_groups[0]["lr"], fold)
            
            # Validation each interval epoch
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler, fold, best_mse)

            # if epoch % self.val_on_epoch == 0:
            #     mse_score_val, _, _, _ = self.eval(fold, epoch=epoch, on_val = not is_full_fold)
            
            mse_score_val, _, _, _ = self.eval(fold, epoch = epoch, on_val = not is_full_fold)
            
            # Save model if achieve the best result on MSE 
            best_mse = min(mse_score_val, best_mse)
            self.exp.delete_model(mse_score_val == best_mse, fold, epoch)

        mse_score_test, ci_score_test, spearman_score_test, pearson_score_test = self.eval(fold=fold, epoch=self.exp.get_last_checkpoint_epoch())
        self.exp.train_end_callback()

        return mse_score_val, mse_score_test, ci_score_test, spearman_score_test , pearson_score_test

    def eval(self,
            fold: int = 6,
            on_val: bool = False,
            epoch: int or None = None) -> tuple:
        model = self.cfg.get_model()
        model.load_state_dict(self.exp.get_epoch_model(epoch, fold))
        model.to(self.device)

        if on_val:            
            dataloader, edge_index_drug_protein = self.get_val_dataloader(fold)
            des = f"Validation phrase on epoch {epoch}"
        else:
            dataloader, edge_index_drug_protein = self.get_test_dataloader()
            des = "Test phrase"
        
        if model == None or dataloader == None or edge_index_drug_protein == None:
            raise Exception("Missing value for running eval function. Please double check them")

        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        model.eval()
        self.exp.eval_start_callback()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=des)
            for data in pbar:
                data_drug, data_protein = map(lambda instance: instance.to(self.device), data)

                if self.cfg.config["model"]["model"] == "SGDTA":
                    new_edge = torch.LongTensor(data_drug.edge_index_drug_protein).transpose(1, 0).to(self.device)
                    edge_index_drug_protein = torch.cat((edge_index_drug_protein, new_edge, new_edge[[1,0]]), 1)
                    output = model(data_drug, data_protein, edge_index_drug_protein)
                else:
                    output = model(data_drug, data_protein)

                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, data[0].y.view(-1, 1).cpu()), 0)
        
            del edge_index_drug_protein

        ground_truth = total_labels.numpy().flatten()
        prediction = total_preds.numpy().flatten()
        mse_score, ci_score, spearman_score, pearson_score = mse(ground_truth, prediction), ci(ground_truth, prediction), spearman(ground_truth, prediction), pearson(ground_truth, prediction)
        self.exp.eval_end_callback(mse_score, ci_score, spearman_score, pearson_score, epoch, on_val, fold)
        return mse_score, ci_score, spearman_score, pearson_score

    def train_per_kfold(self) -> None:
        final_scores_by_folds = []
        for i in range(self.cfg.config["datasets"]["kfold"]):
            self.logger.info(f'Beginning training on {i+1} FOLD session.\n')
            scores = self.train(i+1)
            final_scores_by_folds.append([i] + list(scores))
        
        # get the best score on validation set per fold
        best_fold, *best_scores = max(final_scores_by_folds, key=itemgetter(2))
        self.logger.info(f"Best performance (ci) on fold {best_fold+1}:")
        self.logger.info("MSE val: {:.5f}, MSE test: {:.5f}, CI val: {:.5f}, CI test:{:.5f}.\n".format(*best_scores))
        
    def get_train_dataloader(self, fold: int) -> DataLoader:
        train_dataset = self.cfg.get_dataset(mode='train', fold = fold)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_batch_size,
                                  shuffle=True,
                                  collate_fn=collate,
                                  num_workers=8,
                                  worker_init_fn=self._worker_init_fn_)
        return train_loader, train_dataset.edge_index_drug_protein.to(self.device)

    def get_val_dataloader(self, fold: int) -> DataLoader:
        val_dataset = self.cfg.get_dataset(mode='val', fold = fold)

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_batch_size,
                                shuffle=False,
                                collate_fn=collate,
                                num_workers=8,
                                worker_init_fn=self._worker_init_fn_)
        return val_loader, val_dataset.edge_index_drug_protein.to(self.device)

    def get_test_dataloader(self) -> DataLoader:
        test_dataset = self.cfg.get_dataset(mode='test')

        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=self.test_batch_size,
                                shuffle=False,
                                collate_fn=collate,
                                num_workers=8,
                                worker_init_fn=self._worker_init_fn_)                       
        return test_loader, test_dataset.edge_index_drug_protein.to(self.device)

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)