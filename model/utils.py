import torch
from model.iternet.iternet_classifier import *
from model.efficientnet.EfficientNet import EfficientNet
from model.ResidualAttentionNetwork.residual_attention_network import ResidualAttentionModel_448input as ResidualAttentionModel
import warnings

def get_features_extractor_model(arch, num_classes, features_extractor_path=None, classifier_path=None, freeze=True, pretrained=True):
   
    # initialize the features extractor
    features_extractor = Identity()
    size = 256
    
    if arch == 'iternet_classifier':
        model = Classifier(num_classes=num_classes)
        features_extractor = IternetFeaturesExtractor(path=features_extractor_path)
    elif arch == 'iternet_classifier_data':
        model = ClassifierData(num_classes=num_classes)
        features_extractor = IternetFeaturesExtractor(path=features_extractor_path)
    elif arch == 'iternet_extractor_classifier':
        print('-------Freeze status:', freeze)
        model = IternetClassifier(path=features_extractor_path, num_classes=num_classes, freeze=freeze)
    elif arch == 'iternet_extractor_classifier_data':
        model = IternetClassifierData(path=features_extractor_path, num_classes=num_classes)
    elif 'efficient' in arch:
        if pretrained:
            model = EfficientNet.from_pretrained(arch, num_classes=8)
            path = 'exp/efficientnet-b7-pretraining/efficientnet-b7_Focal_None_epoch_138.pth'
            print('Loading pretrained weights for EfficientNet')
            new_state_dict = load_pretrained_weights(path)
            model.load_state_dict(new_state_dict)
            model._fc = torch.nn.Linear(model._fc.in_features, num_classes)
        else:
            model = EfficientNet.from_pretrained(arch, num_classes=num_classes)
        size = 224
    elif arch == 'residual_attention':
        # load pretrained model
        model = ResidualAttentionModel()
        if pretrained:
            model.fc = torch.nn.Linear(2048, 8)
            path = 'exp/residual-attention-pretraining/residual_attention_Focal_None_epoch_74.pth'
            print('Loading pretrained weights for residual_attention')
            new_state_dict = load_pretrained_weights(path)
            model.load_state_dict(new_state_dict)
        model.fc = torch.nn.Linear(2048, num_classes)
        size = 448
    else:
        warnings.warn('Unknown architecture')
        return
        
    # load classifiers weights
    if classifier_path is not None:
        print('--------Loading Model Weights--------')
        print('--------Model Path--------', classifier_path)
        new_state_dict = load_pretrained_weights(classifier_path)
        model.load_state_dict(new_state_dict)
        print('--------Weights loaded---------')
        
    return features_extractor, model, size