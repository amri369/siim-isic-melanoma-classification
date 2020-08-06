import torchvision.transforms as transforms
from dataset.dataset_melanoma import DatasetMelanoma as Dataset
from torch.utils.data import DataLoader
from model.iternet.iternet_classifier import *
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

class Predict(object):
    def __init__(self, arch_model='Classifier', arch_extractor='IternetFeaturesExtractor', 
                 classifier_path=None, features_extractor_path=None):
        # get the features extractor
        assert arch_extractor in ['Identity', 'IternetFeaturesExtractor'], print('Unknown architecture')
        if arch_extractor == 'Identity':
            features_extractor = Identity()
        elif arch_extractor == 'IternetFeaturesExtractor':
            features_extractor = IternetFeaturesExtractor(path=features_extractor_path)
        
        # get the model
        assert arch_model in ['IternetClassifier', 'Classifier'], print('Unknown architecture')
        if arch_model == 'Classifier':
            model = Classifier(num_classes=2)
        elif arch_model == 'IternetClassifier':
            model = IternetClassifier(path=features_extractor_path, num_classes=2)
        
        # load classifiers weights
        if classifier_path is not None:
            new_state_dict = load_pretrained_weights(classifier_path)
            model.load_state_dict(new_state_dict)
            
        # move to cuda
        self.is_gpu_available = torch.cuda.is_available()
        if self.is_gpu_available:
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
            
    def __call__(self, df, image_dir, target_col='target'):
        # copy the dataframe
        df = df.copy()
        if target_col in df.columns:
            target = True
        else:
            target = False
        
        # set the transformer
        mean = [104.00699, 116.66877, 122.67892]
        std = [0.225*255, 0.224*255, 0.229*255]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # initialize the dataset and the dataloader
        dataset = Dataset(df=df, image_dir=image_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False,
                                num_workers=4, pin_memory=True, sampler=None)
        
        # evaluate
        softmax = torch.nn.Softmax(dim=1)
        all_preds = []
        all_prob = []
        with torch.no_grad():
            for i, ((x, data), y) in enumerate(dataloader):
                # predict
                if self.is_gpu_available:
                    x, data, y = x.cuda(), data.cuda(), y.cuda()
                    
                z = self.features_extractor(x)
                z = self.model(z, data)
                
                # prediction
                _, pred = torch.max(z, 1)
                all_preds.extend(pred.cpu().numpy())
                
                # cancer probability
                prob = softmax(z)
                all_prob.extend(prob[:,1].detach().cpu().numpy())
                print('------', i, '/', len(dataloader))
                
        # calculate ROC if possible
        if target:
            all_targets = df[target_col].tolist()
            roc = roc_auc_score(all_targets, all_prob)
            
            # print confusion matrix and roc
            cm = confusion_matrix(all_targets, all_preds)
            print('ROC:', roc)
            print('CM:', cm)
        
        return all_prob
            