import os
import torch
import torchvision
import torch.nn as nn
import random
import numpy as np
from trainer.evaluate import Evaluate

class Trainer(object):

    def __init__(self, model, criteria, optimizer, scheduler, gpus, seed, writer):
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gpus = gpus
        self.is_gpu_available = torch.cuda.is_available()
        self.seed = seed
        self.writer = writer

    def set_devices(self):
        if self.is_gpu_available:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)
            self.criteria = self.criteria.cuda()
        else:
            self.model = self.model.cpu()
            self.criteria = self.criteria.cpu()
            
    def set_seed(seed):
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        random.seed(seed)

        np.random.seed(seed)

        torch.backends.cudnn.deterministic = True

    def training_step(self, dataloader):
        # initialize the loss
        epoch_loss = 0.0

        # loop over training set
        self.model.train()
        for x, y in dataloader:
            # predict
            if self.is_gpu_available:
                x, y = x.cuda(), y.cuda()
            with torch.set_grad_enabled(True):
                z = self.model(x)
                loss = self.criteria(z, y)
                
            # evaluate
            self.evaluate_train.step(torch.sigmoid(z), y, len(x))
                
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(loss.cpu().data.numpy())

            epoch_loss += loss.item() * len(x)

        epoch_loss = epoch_loss / len(dataloader)
        return epoch_loss
    
    def validation_step(self, dataloader, epoch):
        mean = [104.00699, 116.66877, 122.67892]
        std = [0.225*255, 0.224*255, 0.229*255]
        # initialize the loss
        epoch_loss = 0.0

        # loop over validation set
        self.model.eval()
        for x, y in dataloader:
            # predict
            if self.is_gpu_available:
                x, y = x.cuda(), y.cuda()
            with torch.set_grad_enabled(False):
                z = self.model(x)
                loss = self.criteria(z, y)
                
            # evaluate
            z = torch.sigmoid(z)
            self.evaluate_val.step(z, y, len(x))
            
            epoch_loss += loss.item() * len(x)
            
            # tensorboard
            x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
            x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
            x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
            img_grid = torchvision.utils.make_grid(x)
            gt_grid = torchvision.utils.make_grid(y)
            pred_grid = torchvision.utils.make_grid(z)
            self.writer.add_image('Input images', img_grid, epoch)
            self.writer.add_image('Ground truth', gt_grid, epoch)
            self.writer.add_image('Prediction', pred_grid, epoch)
        
        epoch_loss = epoch_loss / len(dataloader)
        return epoch_loss

    def save_checkpoint(self, epoch, model_dir):
        # create the state dictionary
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_out_path = os.path.join(model_dir, "_epoch_{}.pth".format(epoch))
        torch.save(state, model_out_path)

    def __call__(self, dataloaders, epochs, model_dir):
        Trainer.set_seed(self.seed)
        self.set_devices()
        for epoch in range(epochs):
            # initialize evaluation metrics
            self.evaluate_train = Evaluate()
            self.evaluate_val = Evaluate()
            
            # train and validate
            train_loss = self.training_step(dataloaders['train'])
            val_loss = self.validation_step(dataloaders['val'], epoch)
            self.save_checkpoint(epoch, model_dir)
            
            # log metrics
            print('------', epoch+1, '/', epochs, train_loss, val_loss)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            keys = ['auc', 'acc', 'F1', 'SP', 'SE']
            for key in keys:
                self.writer.add_scalar(key + '/train', self.evaluate_train.scalars[key], epoch)
                self.writer.add_scalar(key + '/val', self.evaluate_val.scalars[key], epoch)
        
        self.writer.close()
            
        if self.is_gpu_available:
                torch.cuda.empty_cache()
