from torch.utils.data import Dataset
import torch
import os
import numpy as np
import pandas as pd
from PIL import Image

meta_data = ['age_approx', 'sex_female', 'sex_male',
             'anatom_site_general_challenge_head/neck',
             'anatom_site_general_challenge_lower extremity',
             'anatom_site_general_challenge_oral/genital',
             'anatom_site_general_challenge_palms/soles',
             'anatom_site_general_challenge_torso',
             'anatom_site_general_challenge_upper extremity']

class DatasetMelanoma(Dataset):
    
    def __init__(self, df, image_dir, img_col='image_name', target_col='target',
                 age_col='age_approx', mean_age=48.,
                 categ_cols=['sex', 'anatom_site_general_challenge'],
                 transform=None, target_transform=None):
        """
        Args:
            df (Pandas dataframe): 
            image_dir (string): Directory with all the images.
            img_col (string): column name containing images names.
            transform (callable, optional): Optional transform to be applied on images only.
        """
        df = impute(df)
        self.annotation = self.transform_df(df, age_col, mean_age, categ_cols)
        self.targets = self.annotation[target_col].values
        self.img_col = img_col
        self.target_col = target_col
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def transform_df(self, df, age_col, mean_age, categ_cols):
        df = df.copy()
        onehot = pd.get_dummies(df[categ_cols])
        df.drop(categ_cols, axis=1, inplace=True)
        df = df.join(onehot)
        df[age_col] = df[age_col] / mean_age
        return df
        
    def get_cls_num_list(self):
        # initialization
        target = self.annotation[self.target_col]
        num_classes = target.max() + 1
        cls_num_list = []
        
        # number of samples per class
        for idx in range(num_classes):
            count = (target == idx).sum()
            cls_num_list.append(count)
            
        return cls_num_list
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        filename = self.annotation[self.img_col].iloc[idx]
        target = self.annotation[self.target_col].iloc[idx]
        data = self.annotation[meta_data].iloc[idx]
        
        # get image data
        img_name = os.path.join(self.image_dir, filename + '.jpg')
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        data = np.array(data, dtype=np.float)
        assert not np.any(np.isnan(data))
        data = torch.from_numpy(data).float()
        
        target = np.asarray(target)
        target = torch.from_numpy(target).long()
        
        return (img, data), target

def impute(df):
    df = df.copy()
    df['sex'].fillna('female', inplace=True)
    df['age_approx'].fillna(df['age_approx'].mean(), inplace=True)
    return df