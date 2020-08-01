import torch
import torch.nn as nn
from .iternet_model import Iternet
from collections import OrderedDict
from .unet_parts import *

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class IdentityBi(nn.Module):
    def __init__(self):
        super(IdentityBi, self).__init__()
        
    def forward(self, x, y):
        return x

class IternetClassifier(nn.Module):
    def __init__(self, num_classes=2, path=None):
        super(IternetClassifier, self).__init__()
        features_extractor = Iternet(n_channels=3, n_classes=1, out_channels=32, iterations=3)
        
        # load pretrained weights
        if path is not None:
            new_state_dict = load_pretrained_weights(path)
            features_extractor.load_state_dict(new_state_dict)
            
        # remove the last upsampling layers
        features_extractor.model_miniunet[-1].up1 = IdentityBi()
        features_extractor.model_miniunet[-1].up2 = IdentityBi()
        features_extractor.model_miniunet[-1].up3 = IdentityBi()
        features_extractor.model_miniunet[-1].outc = Identity()
        self.features_extractor = features_extractor
        
        # add convolutions
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        
        # add a classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.down4(x)
        x = self.down5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_pretrained_weights(path):
    state_dict = torch.load(path, map_location=torch.device('cpu'))['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict