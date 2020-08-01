import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from trainer.utils import *
import time
from sklearn.metrics import confusion_matrix

class Trainer(object):

    def __init__(self, datasets, features_extractor, model, 
                 loss_type, optimizer, lr,batch_size, gpus, workers, seed, writer, 
                 store_name='', resume=None, train_rule=None):
        self.datasets = datasets
        self.cls_num_list = self.datasets['train'].get_cls_num_list()
        self.features_extractor = features_extractor
        self.model = model
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.lr = lr
        self.gpus = gpus
        self.is_gpu_available = torch.cuda.is_available()
        self.seed = seed
        self.writer = writer
        self.store_name = store_name
        self.resume = resume
        self.train_rule = train_rule
        
        # initialize dataloders
        train_sampler = get_sampler(train_rule)
        self.train_loader = DataLoader(
            datasets['train'], batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler)
        self.val_loader = DataLoader(
            datasets['val'], batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, sampler=train_sampler)
        

    def set_devices(self):
        if self.is_gpu_available:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
            self.features_extractor = self.features_extractor.cuda()
            self.model = self.model.cuda()
            self.features_extractor = torch.nn.DataParallel(self.features_extractor)
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.features_extractor = self.features_extractor.cuda()
            self.model = self.model.cpu()

    def training_step(self, epoch):
        end = time.time()
        # get classes weights
        per_cls_weights = get_weights(epoch, self.train_rule, self.datasets['train'])
        if self.is_gpu_available:
            per_cls_weights = per_cls_weights.cuda()
        print('-----per_cls_weights', per_cls_weights)
            
        # set criterion
        self.criterion = get_criteria(self.loss_type, self.cls_num_list, per_cls_weights)
        if self.is_gpu_available:
            self.criterion = self.criterion.cuda()
        
        # initialize metrics
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        # loop over training set
        self.model.train()
        for i, ((x, _), y) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
        
            # mount to GPU
            if self.is_gpu_available:
                x, y = x.cuda(), y.cuda()
                
            # predict
            with torch.set_grad_enabled(True):
                z = self.features_extractor(x)  
                z = self.model(z)
                loss = self.criterion(z, y)
                
            # measure accuracy and record loss
            l = len(y)
            acc1 = accuracy(z, y, topk=(1, ))[0]
            losses.update(loss.item(), l)
                
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          epoch, i, len(self.train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, lr=self.optimizer.param_groups[-1]['lr']))
            print(output)

        return loss.item(), acc1
    
    def validation_step(self, epoch):
        mean = [104.00699, 116.66877, 122.67892]
        std = [0.225*255, 0.224*255, 0.229*255]
        
        # initialize metrics
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        # loop over validation set
        self.model.eval()
        all_preds = []
        all_targets = []
        
        # validate
        end = time.time()
        with torch.no_grad():
            for i, ((x, _), y) in enumerate(self.val_loader):
                # predict
                if self.is_gpu_available:
                    x, y = x.cuda(), y.cuda()

                z = self.features_extractor(x)
                z = self.model(z)
                loss = self.criterion(z, y)

                # measure accuracy and record loss
                l = len(y)
                acc1 = accuracy(z, y, topk=(1, ))[0]
                losses.update(loss.item(), l)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(z, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
                # print loss and accuracy
                output = ('Val: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'
                          .format(i, len(self.val_loader), batch_time=batch_time, loss=losses))
                print(output)
                
                # tensorboard
                x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
                x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
                x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
                img_grid = torchvision.utils.make_grid(x)
                self.writer.add_image('Input images', img_grid, epoch)
                
            # print confusion matrix
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            out_cls_acc = 'Validation: Class Accuracy: %s'%((np.array2string(cls_acc, separator=',', 
                                                                                 formatter={'float_kind':lambda x: "%.3f" % x})))
        
        return loss.item(), acc1

    def save_checkpoint(self, epoch, model_dir):
        # create the state dictionary
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_out_path = os.path.join(model_dir, "store_name_epoch_{}.pth".format(epoch))
        model_out_path = os.path.join(model_dir, self.store_name + "_epoch_{}.pth".format(epoch))
        torch.save(state, model_out_path)
        
    def resume_checkpoint(self):
        resume = self.resume
        if os.path.isfile(self.resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location='cuda:0')
            epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            epoch = 0
        return epoch

    def __call__(self, epochs, model_dir):
        # preparation
        set_seed(self.seed)
        self.set_devices()
        
        # resume checkpoint
        if self.resume is not None:
            start_epoch = self.resume_checkpoint()
        else:
            start_epoch = 0
        
        for epoch in range(start_epoch, epochs):            
            # train and validate
            adjust_learning_rate(self.optimizer, epoch, self.lr)
            train_loss, train_acc = self.training_step(epoch)
            val_loss, val_acc = self.validation_step(epoch)
            self.save_checkpoint(epoch, model_dir)
            
            # log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Acc/train', train_acc, epoch)
            self.writer.add_scalar('Acc/val', val_acc, epoch)
        
        self.writer.close()
            
        if self.is_gpu_available:
                torch.cuda.empty_cache()
