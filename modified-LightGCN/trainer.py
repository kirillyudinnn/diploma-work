import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn

from typing import Dict

from loss import SampledSoftmaxLoss

class Trainer:
    def __init__(self, CONFIG: Dict):
        self.config = CONFIG
        self.n_epochs = CONFIG["n_epochs"]
        self.train_batch_size = CONFIG["train_batch_size"]
        self.val_batch_size = CONFIG["val_batch_size"]
        self.lambda_ = CONFIG["lambda_"]
        self.device = torch.device(CONFIG["device"])
        self.opt_fn = lambda model: Adam(
            model.parameters(), CONFIG["lr"]
        )
        self.scheduler_fn = lambda optimizer: ReduceLROnPlateau(optimizer, **CONFIG["scheduler"])
        self.loss = SampledSoftmaxLoss(**CONFIG["loss"])
        self.verbose = CONFIG["verbose"]
        self.scheduler = None
        self.save_best_embs = CONFIG["save_best_embs"]
        self.history = {}
        self.model = None
        self.E0_best = None
        self.best_epoch = -1


    def fit(self, model, train_loader, val_loader):
        """
            Fit model 
            param model: model
            param train_loader: train data loader
            param val_loader: validation data loader
        """

        self.model = model
        self.optimizer = self.opt_fn(model)
        self.scheduler = self.scheduler_fn(self.optimizer)

        train_losses = []
        val_losses = []
        best_val_loss = 1000000.0
        EPOCHS = range(self.n_epochs)

        for epoch in EPOCHS:
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)

            val_loss = self._val_epoch(val_loader)
            val_losses.append(val_loss)

            if self.save_best_embs:
                if (val_loss < best_val_loss) and ((val_loss - best_val_loss > 0.005) or ((val_loss - best_val_loss < 0))):
                    best_val_loss = val_loss
                    self.E0_best = self.model.state_dict()["E0.weight"].cpu().numpy()
                    self.best_epoch = epoch

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            

        self.history["train_loss"] = train_losses
        self.history["val_loss"] = val_losses



        self.model.E0 = nn.Embedding.from_pretrained(torch.Tensor(self.E0_best))
        model.E0.to(self.device)
        
        return self.model.eval()

    def _train_epoch(self, train_loader):
        """
            Training epoch with random users batch.
            param train_loader: train data loader 
        """
        self.model.train()
        n_users = train_loader.n_users

        n_batch = n_users // self.train_batch_size

        avg_train_loss = 0.0

        for batch in range(n_batch):
            self.optimizer.zero_grad()

            batch_users = np.random.choice(
                a=n_users,
                size=self.train_batch_size, 
                replace=False
            )

            users, pos_items, neg_items = train_loader[batch_users]

            main_loss, reg_loss = self._forward_n_loss(users, pos_items, neg_items)
            final_loss = main_loss + reg_loss
            avg_train_loss += final_loss.item()

            final_loss.backward()
            self.optimizer.step()

        avg_train_loss /= n_batch

        return avg_train_loss
    
    def _val_epoch(self, val_loader):
        """
            Evaluate val loss
            param data_loader: valildation data loader
        """

        self.model.eval()
        avg_val_loss = 0.0
        n_users = val_loader.n_users
        n_batch = n_users // self.val_batch_size

        for batch in range(n_batch):
            batch_users = np.arange(
                start=batch*self.val_batch_size,
                stop=(batch+1)*self.val_batch_size
            )

            users, pos_items, neg_items = val_loader[batch_users]

            with torch.no_grad():
                main_loss, reg_loss = self._forward_n_loss(users, pos_items, neg_items)
                final_loss = main_loss + reg_loss
                avg_val_loss += final_loss.item()

        avg_val_loss /= n_batch

        return avg_val_loss

    
    def _forward_n_loss(
            self,
            users: np.array, 
            pos_items: np.array, 
            neg_items: np.array
    ):
        """
            Compute loss 
        """

        usr_embeds, pos_embeds, neg_embeds = self.model.forward(users, pos_items, neg_items)

        main_loss = self.loss.forward(usr_embeds, pos_embeds, neg_embeds)
        reg_loss = self.lambda_ * self.model.E0.weight.pow(2).sum()

        return main_loss, reg_loss
