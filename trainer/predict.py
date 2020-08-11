import torchvision.transforms as transforms
from dataset.dataset_melanoma import DatasetMelanoma as Dataset
from torch.utils.data import DataLoader
from model.utils import get_features_extractor_model
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import os

class Predict(object):
    def __init__(self, arch='iternet_extractor_classifier_data', 
                 classifier_path=None, 
                 features_extractor_path=None,
                 is_gpu_available=True):
            
        # get model and features extractor
        features_extractor, model, size = get_features_extractor_model(arch, num_classes=2, 
                                                                       features_extractor_path=features_extractor_path, 
                                                                       classifier_path=classifier_path, 
                                                                       freeze=True)
        
        # move to cuda
        self.is_gpu_available = is_gpu_available
        if self.is_gpu_available:
            torch.backends.cudnn.benchmark = True
            #os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,11,14'
            features_extractor = features_extractor.cuda()
            model = model.cuda()
            features_extractor = torch.nn.DataParallel(features_extractor)
            model = torch.nn.DataParallel(model)
            
        # set eval mode
        model = model.eval()
        features_extractor = features_extractor.eval()
        
        # update the self
        self.model = model
        self.features_extractor = features_extractor
            
    def __call__(self, df, image_dir, iterations=10, augmentation=None, target_col='target', validate=True, size=None):
        # copy the dataframe
        df = df.copy()
        if target_col in df.columns:
            target = True
        else:
            target = False
        
        # set the transformer
        mean = [104.00699, 116.66877, 122.67892]
        std = [0.225*255, 0.224*255, 0.229*255]
        if augmentation is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            iterations = 1
            print('-----Without data augmentation-----')
        else:
            print('-----With data augmentation-----')
            transform = transforms.Compose([
                augmentation,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
        if size is not None:
            print('-----Resizing to:', size)
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transform
            ])
            
        
        # initialize the dataset and the dataloader
        dataset = Dataset(df=df, image_dir=image_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False,
                                num_workers=4, pin_memory=True, sampler=None)
        
        # evaluate
        softmax = torch.nn.Softmax(dim=1)
        all_prob_dic = {}
        for k in range(iterations):
            print('-----Iteration-----', k)
            all_prob = []
            with torch.no_grad():
                for i, ((x, data), y) in enumerate(dataloader):
                    # predict
                    if self.is_gpu_available:
                        x, data, y = x.cuda(), data.cuda(), y.cuda()

                    z = self.features_extractor(x)
                    z = self.model(z, data)

                    # cancer probability
                    prob = softmax(z)
                    all_prob.extend(prob[:,1].detach().cpu().numpy())
                    print('------', i, '/', len(dataloader))
            all_prob_dic[i] = all_prob
        
        if self.is_gpu_available:
            torch.cuda.empty_cache()
            
        # average predictions
        all_prob_dic = pd.DataFrame(all_prob_dic)
        all_prob = all_prob_dic.mean(axis=1).values
        
        # maximum vote
        all_pred = (all_prob >= 0.5) * 1
                
        # calculate ROC if possible
        if validate:
            all_targets = df[target_col].tolist()
            roc = roc_auc_score(all_targets, all_prob)
            
            # print confusion matrix and roc
            cm = confusion_matrix(all_targets, all_pred)
            print('ROC:', roc)
            print('CM:', cm)
        
        return all_prob
            