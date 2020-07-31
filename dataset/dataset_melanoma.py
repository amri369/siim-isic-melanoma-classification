from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image

class DatasetMelanoma(Dataset):
    
    def __init__(self, df, image_dir, img_col='image_name',
                 target_col='target', transform=None, target_transform=None):
        """
        Args:
            df (Pandas dataframe): 
            image_dir (string): Directory with all the images.
            img_col (string): column name containing images names.
            transform (callable, optional): Optional transform to be applied on images only.
        """
        self.annotation = df[[img_col, target_col]]
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        filename, target = self.annotation.iloc[idx]
        
        # get image data
        img_name = os.path.join(self.image_dir, filename + '.jpg')
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target = np.asarray(target)
        target = torch.from_numpy(target).long()
        
        return img, target
