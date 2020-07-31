import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from losses import LDAMLoss, FocalLoss
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_melanoma import DatasetMelanoma as Dataset
from augmentation.autoaugment import ImageNetPolicy
import torchvision.transforms as transforms
from model.iternet.iternet_classifier import IternetClassifier
from trainer.trainer import Trainer
import pandas as pd

import argparse

def main(args): 
    #
    mean = [104.00699, 116.66877, 122.67892]
    std = [0.225*255, 0.224*255, 0.229*255]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    Transform = {'train': transform, 'val': transform}
    
    # set datasets
    dataframe = {
        'train': pd.read_csv(args.train_csv),
        'val': pd.read_csv(args.val_csv)
    }
    
    datasets = {
        x: Dataset(df=dataframe[x],
                   image_dir=args.image_dir,
                   transform=transform) for x in ['train', 'val']
    }

    # set dataloaders
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True) for x in ['train', 'val']
    }

    # initialize the model
    model = IternetClassifier(num_classes=2, path=args.pretrained_checkpoint)
        
    # freeze the features extractor layers
    if args.freeze:
        for param in model.features_extractor.parameters():
            param.requires_grad = False

    # set loss function and optimizer
    criteria = FocalLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30])
    
    #
    writer = SummaryWriter('tensorboard/' + args.arch)
    # train the model
    trainer = Trainer(model, criteria, optimizer,
                      scheduler, args.gpus, args.seed, writer, args.resume)
    exp = os.path.join(args.model_dir, args.arch)
    trainer(dataloaders, args.epochs, exp)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--gpus', default='4,5,6,7',
                        type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='iternet',
                        type=str, help='Architecture')
    parser.add_argument('--pretrained_checkpoint', default='../cancer-segmentation/exp/iternet/_epoch_99.pth', 
                        type=str, help='path to the pretrained model')
    parser.add_argument('--freeze', action='store_true', 
                        help='Freeze network except last layer')
    parser.add_argument('--resume', default='', 
                        type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--size', default='256', type=int,
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--image_dir', default='data/jpeg/train/',
                        type=str, help='Images folder path')
    parser.add_argument('--train_csv', default='data/train_split.csv',
                        type=str, help='list of training set')
    parser.add_argument('--val_csv', default='data/val_split.csv',
                        type=str, help='list of validation set')
    parser.add_argument('--lr', default='0.0001',
                        type=float, help='learning rate')
    parser.add_argument('--epochs', default='2',
                        type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default='32',
                        type=int, help='Batch Size')
    parser.add_argument('--model_dir', default='exp/',
                        type=str, help='Images folder path')
    parser.add_argument('--seed', default='2020123',
                        type=int, help='Random status')
    args = parser.parse_args()

    main(args)
