from model.iternet.iternet_classifier import *
from model.efficientnet.EfficientNet import EfficientNet
import warnings

def get_features_extractor_model(arch, num_classes, features_extractor_path=None, classifier_path=None, freeze=True):
    print('-------Freeze status:', freeze)
   
    # initialize the features extractor
    features_extractor = Identity()
    
    if arch == 'iternet_classifier':
        model = Classifier(num_classes=num_classes)
        features_extractor = IternetFeaturesExtractor(path=features_extractor_path)
    elif arch == 'iternet_classifier_data':
        model = ClassifierData(num_classes=num_classes)
        features_extractor = IternetFeaturesExtractor(path=features_extractor_path)
    elif arch == 'iternet_extractor_classifier':
        model = IternetClassifier(path=features_extractor_path, num_classes=num_classes, freeze=freeze)
    elif arch == 'iternet_extractor_classifier_data':
        model = IternetClassifierData(num_classes=num_classes, freeze=freeze)
    elif 'efficient' in arch:
        model = EfficientNet.from_pretrained(arch, num_classes=num_classes)
    else:
        warnings.warn('Unknown architecture')
        return
        
    # load classifiers weights
    if classifier_path is not None:
        print('--------Loading Model Weights--------')
        new_state_dict = load_pretrained_weights(classifier_path)
        model.load_state_dict(new_state_dict)
        print('--------Weights loaded---------')
        
    return features_extractor, model