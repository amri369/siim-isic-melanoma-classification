from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np
# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)
    corr = np.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == np.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    SE = float(np.sum(TP))/(float(np.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(np.sum(TN))/(float(np.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(np.sum(TP))/(float(np.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == np.max(GT)
    
    Inter = np.sum((SR+GT)==2)
    Union = np.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == np.max(GT)

    Inter = np.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(np.sum(SR)+np.sum(GT)) + 1e-6)

    return DC

def get_auc(SR,GT):
    SR = SR.flatten().detach().cpu().numpy()
    GT = GT.flatten().detach().cpu().numpy().astype(int)
    fpr, tpr, _ = roc_curve(GT, SR)
    AUC = auc(fpr, tpr)
    
    return AUC

class Evaluate:
    
    def __init__(self):
        # metrics
        self.f = {'acc': get_accuracy, 'SE': get_sensitivity, 
                  'SP': get_specificity, 'PC': get_precision, 
                  'F1': get_F1, 'JS': get_JS, 
                  'DC': get_DC, 'auc': get_auc}
        
        # scalars
        self.scalars = {}
        for key in self.f:
            self.scalars[key] = 0.
        
    def step(self, SR, GT, length):
        
        # update
        for key in self.f:
            self.scalars[key] += self.f[key](SR,GT) / length

def get_metrics(all_targets, all_prob):
    metrics = {}
    all_targets, all_prob = np.array(all_targets), np.array(all_prob)
    all_pred = (all_prob >= 0.5) * 1
    try:
        metrics['ROC'] = roc_auc_score(all_targets, all_prob)
    except:
        metrics['ROC'] = -1
    return metrics