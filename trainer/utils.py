import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import numpy as np
import random
from losses import LDAMLoss, FocalLoss
from sklearn.metrics import confusion_matrix
import warnings

def get_sampler(train_rule, dataset):
    if train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(dataset)
    else:
        train_sampler = None        
    return train_sampler
    
    
def get_weights(epoch, train_rule, dataset, k=1):
    # get number of samples per class
    cls_num_list = dataset.get_cls_num_list()

    # get sampler and weights
    if train_rule == 'None':
        per_cls_weights = [1] * len(cls_num_list)
    elif train_rule == 'Resample':
        per_cls_weights = np.array([1, 1])
        per_cls_weights = per_cls_weights / np.linalg.norm(per_cls_weights, ord=2) 
    elif train_rule == 'Reweight':
        #beta = 0.9999
        #effective_num = 1.0 - np.power(beta, cls_num_list)
        #per_cls_weights = (1.0 - beta) / np.array(effective_num)
        #per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = np.array([1, 5])
        per_cls_weights = per_cls_weights / np.linalg.norm(per_cls_weights, ord=2) 
    elif train_rule == 'DRW':
        idx = min(1, epoch // 20)
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    elif train_rule == 'Power':
        cls_num_list = np.array(cls_num_list, dtype=np.float) + 1E-4
        per_cls_weights = cls_num_list.sum() / cls_num_list
        per_cls_weights = np.power(per_cls_weights, k)
    else:
        warnings.warn('Sample rule is not listed')
        return
    
    per_cls_weights = torch.FloatTensor(per_cls_weights)
    
    return per_cls_weights

def get_criteria(loss_type, cls_num_list, per_cls_weights):
    if loss_type == 'CE':
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights)
    elif loss_type == 'LDAM':
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights)
    elif loss_type == 'Focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=1)
    else:
        warnings.warn('Loss type is not listed')
        return
    
    return criterion
    

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #epoch = epoch + 1
    #if epoch <= 10:
    #    lr = lr * epoch / 10
    #elif epoch >= 160:
    #    lr = lr * 0.001
    #elif epoch > 40:
    #    lr = lr * 0.01
    #elif epoch > 20:
    #    lr = lr * 0.1
    #elif epoch > 10:
    #    lr = lr * 0.1
    #else:
    #    lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def set_seed(seed):
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        random.seed(seed)

        np.random.seed(seed)
        
        torch.backends.cudnn.deterministic = True
        
class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def get_class_accuracy(all_targets, all_preds):
    cf = confusion_matrix(all_targets, all_preds)
    return cf

def print_cm(cm, labels=[0, 1], hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print( "    " + empty_cell,)
    for label in labels: 
        print( "%{0}s".format(columnwidth) % label,)
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print( "    %{0}s".format(columnwidth) % label1,)
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print( cell,)
        print()
        
class ImbalancedDatasetSampler_(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
        