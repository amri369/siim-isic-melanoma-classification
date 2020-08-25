from trainer.predict import Predict
import pandas as pd
import os
import argparse
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

class Ensemble(object):
    def __init__(self, params):
        # get learners with their trained weights
        self.params = params
    
    def predict(self, df, image_dir, validate=False, is_gpu_available=True):
        # initialization
        params = self.params
        archs = params['archs']
        paths = params['paths']
        sizes = params['sizes']
        names = params['names']
        
        # loop over predictors
        predictions = {}
        for arch, path, size, name in zip(archs, paths, sizes, names):
            # get predictor
            print('________Prediction for_____', arch, 'under name', name)
            predict = Predict(arch=arch, 
                              classifier_path=path, 
                              features_extractor_path=None, 
                              is_gpu_available=is_gpu_available)
            
            # predict without data augmentation
            all_probs = predict(df, image_dir, validate=validate, size=size)
            predictions[name] = all_probs
        
        del predict
        
        predictions = pd.DataFrame(predictions)
        
        # add image names and targets
        predictions['image_name'] = df['image_name']
        try:
            predictions['target'] = df['target']
        except:
            predictions['target'] = -1
        columns = ['image_name'] + names + ['target']
        predictions = predictions[columns]
        
        return predictions
    
    @staticmethod
    def assemble(predictions, names, method='voting', weights=None):
        print('------Method------', method)
        df = predictions.copy()
        if method == 'voting':
            if weights is None:
                df['prediction'] = df[names].mean(axis=1)
            else:
                s = 0. * df[names[0]]
                for name, weight in zip(names, weights):
                    print(name, weight)
                    s = s + weight * df[name]
                df['prediction'] = s / s.max()
                    
        if method == 'rankdata':
            for name in names:
                df[name] = rankdata(df[name], method='average')
                if weights is None:
                    df['prediction'] = df[names].sum(axis=1)
                    df['prediction'] = df['prediction'] / df['prediction'].max()
                else:
                    s = 0. * df[names[0]]
                    for name, weight in zip(names, weights):
                        print(name, weight)
                        s = s + weight * df[name]
                        df[name] = df[name] / df[name].max()
                    df['prediction'] = s / s.max()
            
        return df

def main(parser):
    # hard coded params
    #params = {}
    #params['archs'] = [
    #    'efficientnet-b7', 'efficientnet-b7', 'efficientnet-b7', 
    #    'iternet_extractor_classifier_data', 'iternet_extractor_classifier_data', 'iternet_extractor_classifier_data',
    #    'residual_attention', 'residual_attention', 'residual_attention'
    #]
    #params['paths'] = [
    #    'exp/best-models/efficientnet-b7-K1-pretrained-ISIC2019.pth',
    #    'exp/best-models/efficientnet-b7-K2-pretrained-ISIC2019.pth',
    #    'exp/best-models/efficientnet-b7-K3-pretrained-ISIC2019.pth',
    #    'exp/best-models/iternet-K-1.pth',
    #    'exp/best-models/iternet-K-2.pth',
    #    'exp/best-models/iternet-K-3.pth',
    #    'exp/best-models/residual_attention-K-1.pth',
    #    'exp/best-models/residual_attention-K-2.pth',
    #    'exp/best-models/residual_attention-K-3.pth',
    #]
    #params['sizes'] = [224, 224, 224, 
    #                   256, 256,
    #                   448, 448, 448
    #                  ]
    #params['names'] = ['efficientnet-b7-K-1', 'efficientnet-b7-K-2', 'efficientnet-b7-K-3', 
    #                   'iternet-K-1', 'iternet-K-2',
    #                   'residual_attention-K-1', 'residual_attention-K-2', 'residual_attention-K-3'
    #                  ]
    
    # only efficient-nets
    params = {}
    params['archs'] = ['efficientnet-b7', 'efficientnet-b7', 'efficientnet-b7',
                       'iternet_extractor_classifier_data', 'iternet_extractor_classifier_data',
                       'iternet_extractor_classifier_data',
                       'residual_attention', 'residual_attention', 'residual_attention'
                      ]
    params['paths'] = [
        'exp/best-models/efficientnet-b7-K1-pretrained-ISIC2019.pth',
        'exp/best-models/efficientnet-b7-K2-pretrained-ISIC2019.pth',
        'exp/best-models/efficientnet-b7-K3-pretrained-ISIC2019.pth',
        'exp/best-models/iternet-K-1.pth',
        'exp/best-models/iternet-K-2.pth',
        'exp/best-models/iternet-K-3.pth',
        'exp/best-models/residual_attention-K-1.pth',
        'exp/best-models/residual_attention-K-2.pth',
        'exp/best-models/residual_attention-K-3.pth'
    ]
    params['sizes'] = [224, 224, 224,
                       256, 256, 256,
                       448, 448, 448
                      ]
    params['names'] = ['efficientnet-b7-K-1', 'efficientnet-b7-K-2', 'efficientnet-b7-K-3',
                      'iternet-K-1', 'iternet-K-2', 'iternet-K-3',
                      'residual_attention-K-1', 'residual_attention-K-2', 'residual_attention-K-3']
    
    # read dataframe
    df = pd.read_csv(args.csv_path)
    if not args.validate:
        df['target'] = 0
    else:
        target = df['target'].tolist()
    
    # predict using the ensemble method
    ensemble = Ensemble(params)
    predictions = ensemble.predict(df, args.image_dir, validate=args.validate, is_gpu_available=args.use_gpu)
    df = Ensemble.assemble(predictions, params['names'], method=args.ensemble_method, weights=None)
    
    # save predictions
    df.to_csv(os.path.join(args.results_path, 'results.csv'), index=None)
    df['target'] = df['prediction']
    df = df[['image_name', 'target']]
    df.to_csv(os.path.join(args.results_path, 'sample_submission.csv'), index=None)
    
    if args.validate:
        roc = roc_auc_score(target, df['target'].tolist())
        print('Final ROC', roc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--csv_path', default="data/val_split.csv", 
                        type=str, help='path of csv file')
    parser.add_argument('--results_path', default="mohamed/submission", 
                        type=str, help='loss type')
    parser.add_argument('--image_dir', default='data/resized-jpeg/train/',
                        type=str, help='Images folder path')
    parser.add_argument('--ensemble_method', default='voting',
                        type=str, help='ensemble method')
    parser.add_argument('--validate', action='store_true', 
                        help='if to validate target')
    parser.add_argument('--use_gpu', action='store_true', 
                        help='use gpu')
    
    args = parser.parse_args()
    
    main(args)
    