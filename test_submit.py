import sys
sys.path.insert(0, "/raid/mohamed/siim-isic-melanoma-classification")
import pandas as pd
from trainer.predict import Predict
from augmentation.autoaugment import *

arch_model = 'iternet_classifier_data'
arch_extractor = 'Identity' 
classifier_path = '/raid/mohamed/siim-isic-melanoma-classification/exp/2020-08-08_16_09_30_095778_lr_1e-05_Focal_Resample_Geometry/Focal_Resample_epoch_81.pth'
features_extractor_path = '/raid/mohamed/cancer-segmentation/exp/iternet/_epoch_99.pth'
predict = Predict(arch_model, arch_extractor, classifier_path, features_extractor_path, is_gpu_available=True)

# test on validation set
df_path = '/raid/mohamed/siim-isic-melanoma-classification/data/val_split.csv'
image_dir = '/raid/mohamed/siim-isic-melanoma-classification/data/resized-jpeg/train/'
df = pd.read_csv(df_path)
all_prob = predict(df, image_dir, iterations=5, augmentation=None)

# test on submission set
df_path = '/raid/mohamed/siim-isic-melanoma-classification/data/test.csv'
image_dir = '/raid/mohamed/siim-isic-melanoma-classification/data/resized-jpeg/test/'
df = pd.read_csv(df_path)
df['target'] = 0

# predict
all_prob = predict(df, image_dir, iterations=5, augmentation=None, validate=False)

# save
df['target'] = all_prob
df = df[['image_name', 'target']]
df.to_csv('/raid/mohamed/siim-isic-melanoma-classification/mohamed/submission/sample_submission.csv', index=None)