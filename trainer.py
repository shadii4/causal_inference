
from modeling import ClinicalCausalBert

import numpy as np
from box import Box
from tqdm import tqdm
import os
import abc
import torch
import tqdm.auto
from typing import Optional
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, DataLoader, Dataset, TensorDataset

class ModelTrainer(abc.ABC):
    """
    Trainer for base models.
    """
    def __init__(
        self,
        model: ClinicalCausalBert,
        optimizer: Optimizer = None,
        loss_fn: nn.Module = None,
        device: Optional[torch.device] = None,
        n_epochs: int = 0,
        X=None,y_propensity=None, y_outcome=None,
        checkpoint_folder: str = None,
        args: Box = None
    ):
        """
        Initialize the trainer.
        :param model: Instance of the bert-like model to train.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        super().__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prop_loss_fn = nn.CrossEntropyLoss()
        self.n_epochs = n_epochs
        self.X, self.y_propensity, self.y_outcome = X,y_propensity, y_outcome
        self.checkpoint_folder = checkpoint_folder
        self.model_name = "clinical-bert-two-layers"
        self.args = args
        train_dataset = TensorDataset(self.y_propensity, self.y_outcome)
        self.dl_train = DataLoader(train_dataset, batch_size=1)
    def save_checkpoint(self, checkpoint_filename: str, epoch_idx: int):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        torch.save(self.model, checkpoint_filename)
        print(f"\n*** epoch = {epoch_idx} Saved checkpoint {checkpoint_filename}")
    def train(self):

        for epoch_idx in range(self.n_epochs):
            self.model.model.train()
            self.model.propensity_estimator.train()
            self.model.outcome_estimator.train()

            print(f'Epoch {epoch_idx + 1:04d} / {self.n_epochs:04d}', end='\n=================\n')

            for batch_idx, (input_text,(y_prop,y_out)) in enumerate(tqdm.tqdm(zip(self.X,self.dl_train),total=len(self.dl_train),desc=f'Epoch = {epoch_idx+1}')):

                if self.device:
                    input_text,y_prop,y_out = input_text,y_prop.to(self.device),y_out.to(self.device)
                propensity_probs, outcome = self.model.forward(input_text,y_prop)
                propensity_probs = propensity_probs.unsqueeze(0)

                # Compute Loss

                prop_loss = self.prop_loss_fn(propensity_probs,y_prop)

                outcome_loss = self.loss_fn(outcome,y_out)
                total_loss = prop_loss + outcome_loss

                # Backward pass
                self.optimizer.zero_grad()  # Zero gradients of all parameters
                total_loss.backward()  # Run backprop algorithms to calculate gradients

                # Optimization step
                self.optimizer.step()

            checkpoint_filename: str = os.path.join(self.checkpoint_folder, self.model_name)
            self.save_checkpoint(checkpoint_filename, (epoch_idx+1))
    def estimate(self,t1_num):
        res = 0
        propensity, outcomesT0, outcomesT1 = [0]*len(self.dl_train),[0]*len(self.dl_train),[0]*len(self.dl_train)
        for batch_idx, (input_text, (y_prop, y_out)) in enumerate(
                tqdm.tqdm(zip(self.X, self.dl_train), total=len(self.dl_train), desc=f'Estimate')):
            
            if self.device:
                input_text, y_prop, y_out = input_text, y_prop.to(self.device), y_out.to(self.device)
            with torch.no_grad():
                propensity_scores, outcome_score_T0 = self.model.forward(input_text,torch.LongTensor([0]))
                _, outcome_score_T1 = self.model.forward(input_text, torch.LongTensor([1]))
            propensity_probs = torch.exp(propensity_scores)/(torch.exp(propensity_scores).sum())
            res += (outcome_score_T1-outcome_score_T0)*propensity_probs[1]
            propensity[batch_idx], outcomesT0[batch_idx], outcomesT1[batch_idx] = propensity_probs[1].detach().item(), outcome_score_T0.detach().item(), outcome_score_T1.detach().item()
        res /= t1_num
        return res , propensity, outcomesT0, outcomesT1