from torch.utils.data import Dataset
import torch
import os
import pandas as pd  
from PIL import Image

class Dataset(Dataset):
    
    def __init__(self, csv_file, image_dir, mask_dir, img_col='image',
                 mask_col='mask', transform=None, batch_size=32):
        """
        Args:
            csv_file (Pandas dataframe): Path to the csv file with list of images in the dataset.
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            col_filename (string): column name containing images names.
            transform_img (callable, optional): Optional transform to be applied on images only.
            transform_img_mask (callable, optional): Optional transform to be applied on images and masks simultaneously.
        """
        self.image_names = pd.read_csv(csv_file)[img_col]
        self.mask_names = pd.read_csv(csv_file)[mask_col]
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get image data
        filename = self.image_names.iloc[idx]
        img_name = os.path.join(self.image_dir, filename)
        img = Image.open(img_name).convert('RGB')
        
        # get mask_data
        filename = self.mask_names.iloc[idx]
        mask_name = os.path.join(self.mask_dir, filename)
        mask = Image.open(mask_name).convert('RGB')
            
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        
        return img, mask