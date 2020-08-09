import os
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_melanoma import DatasetMelanoma as Dataset
from augmentation.autoaugment import *
import torchvision.transforms as transforms
from model.utils import get_features_extractor_model
from trainer.trainer import Trainer
import pandas as pd
from datetime import datetime

import argparse

def main(args):
    # get model and features extractor
    features_extractor, model, size = get_features_extractor_model(args.arch, num_classes=2, 
                                                             features_extractor_path=args.features_extractor_resume, 
                                                             classifier_path=None, 
                                                             freeze=not args.unfreeze)
    
    # get the augmentation strategy
    augmentation = {
        'ImageNetPolicy': ImageNetPolicy(),
        'CIFAR10Policy': CIFAR10Policy(),
        'SVHNPolicy': SVHNPolicy(),
        'Geometry': Geometry(),
        'GeometryContrast': GeometryContrast()
    }
    augmentation = augmentation[args.augmentation]
    
    
    # data transformer
    mean = [104.00699, 116.66877, 122.67892]
    std = [0.225*255, 0.224*255, 0.229*255]
    transform_train = transforms.Compose([
        augmentation,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])
    
    # resizer if necessary
    if size != 256:
        print('-----Image resized to', size)
        transform_train = transforms.Compose([
            transforms.Resize((size, size)),
            transform_train
        ])
        transform_val = transforms.Compose([
            transforms.Resize((size, size)),
            transform_val
        ])
    
    # put transform in one dict
    Transform = {
        'train': transform_train, 
        'val': transform_val
    }
    
    # set datasets
    dataframe = {
        'train': pd.read_csv(args.train_csv),
        'val': pd.read_csv(args.val_csv)
    }
    
    datasets = {
        x: Dataset(df=dataframe[x],
                   image_dir=args.image_dir,
                   transform=Transform[x]) for x in ['train', 'val']
    }
    
    # set optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    
    # initialize tensorboard writer
    now = str(datetime.now()).replace(" ", "_").replace(":", "_")
    experiment_type = args.arch + '_lr_' + str(args.lr) + '_' + args.loss_type + '_' + args.train_rule + '_' + args.augmentation + '_' + now
    experiment_type = experiment_type.replace(".", "_")
    writer = SummaryWriter('tensorboard/' + experiment_type)
    
    # log hyperparameters
    dic = args.__dict__
    i = 1
    for key in dic:
        writer.add_text(key, str(dic[key]), i)
        i += 1
    
    # initialize store_name
    store_name = '_'.join([args.arch, args.loss_type, args.train_rule])
    store_name = store_name.replace(".", "_")
    writer.add_text('Models dir', args.model_dir, 0)
    try:
        os.mkdir(args.model_dir)
    except:
        pass
    
    # initialize a training instance
    trainer = Trainer(datasets, model, features_extractor,
                      args.loss_type, optimizer=optimizer, lr=args.lr, 
                      batch_size=args.batch_size, gpus=args.gpus, 
                      workers=args.workers, seed=args.seed, writer=writer, 
                      store_name=store_name, resume=args.resume, train_rule=args.train_rule)
    
    # train the model
    trainer(args.epochs, args.model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--loss_type', default="Focal", 
                        type=str, help='loss type')
    parser.add_argument('--unfreeze', action='store_true', default=False, 
                        help='freeze layers')
    parser.add_argument('--train_rule', default='DRW', 
                        type=str, help='data sampling strategy for train loader')
    parser.add_argument('--augmentation', default='Geometry', 
                        type=str, help='Augmentation strategy')
    parser.add_argument('--gpus', default='0,1,2,3,4',
                        type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='iternet',
                        type=str, help='Architecture')
    parser.add_argument('--features_extractor_resume', default='../cancer-segmentation/exp/iternet/_epoch_99.pth', 
                        type=str, help='path to the pretrained model')
    parser.add_argument('--resume', default='', 
                        type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--image_dir', default='data/resized-jpeg/train/',
                        type=str, help='Images folder path')
    parser.add_argument('--train_csv', default='data/train_split.csv',
                        type=str, help='list of training set')
    parser.add_argument('--val_csv', default='data/val_split.csv',
                        type=str, help='list of validation set')
    parser.add_argument('--lr', default='0.001',
                        type=float, help='learning rate')
    parser.add_argument('--epochs', default='200',
                        type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default='256',
                        type=int, help='Batch Size')
    parser.add_argument('--model_dir', default='exp/',
                        type=str, help='Images folder path')
    parser.add_argument('--seed', default='2020123',
                        type=int, help='Random status')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    args = parser.parse_args()

    main(args)
