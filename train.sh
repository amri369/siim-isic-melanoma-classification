python main.py --resume exp/iternet/Focal_Resample_epoch_199.pth --train_rule Reweight --epochs 300
python main.py --resume exp/iternet/Focal_Resample_epoch_171.pth --train_rule Reweight --epochs 300
python main.py --train_rule Resample --batch_size 32 --augmentation ImageNetPolicy --epochs 200
python main.py --train_rule Resample --batch_size 32 --augmentation Geometry --epochs 200
python main.py --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 300
python main.py --train_rule None --batch_size 256 --augmentation Geometry --epochs 300 --resume exp/best_models/Focal_Resample_epoch_114.pth
python main.py --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 300 --arch iternet_data
python main.py --train_rule Resample --batch_size 64 --augmentation Geometry --epochs 300 --arch iternet_data --train_csv data/train_split.csv --resume exp/2020-08-06_10_53_17_046937_lr_0_001_Focal_Resample_Geometry/Focal_Resample_epoch_156.pth

kaggle competitions submit -c siim-isic-melanoma-classification -f sample_submission.csv -m "Message"

python main.py --train_rule Power --batch_size 256 --augmentation GeometryContrast --epochs 300 --arch iternet_data 

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 400 --lr 0.001 --arch iternet_data --resume exp/2020-08-07_05_48_12_199703_lr_0_001_Focal_Power_GeometryContrast/Focal_Power_epoch_299.pth

CUDA_VISIBLE_DEVICES=9,10 python main.py --gpus 9,10 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 300 --arch iternet_data --train_csv data/train.csv --resume exp/2020-08-06_10_53_17_046937_lr_0_001_Focal_Resample_Geometry/Focal_Resample_epoch_156.pth

CUDA_VISIBLE_DEVICES=11,12,13,14,15 python test_submit.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python main.py --gpus 0,1,2,3,4,5,6 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 300 --arch iternet_classifier_data --unfreeze =- usefull

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python main.py --lr 0.00001 --gpus 0,1,2,3,4,5,6 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 150 --arch iternet_classifier_data --unfreeze --train_csv data/train.csv --resume exp/2020-08-08_12_03_05_586405_lr_0_001_Focal_Resample_Geometry/Focal_Resample_epoch_80.pth -= useless

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python main.py --gpus 0,1,2,3,4,5,6 --train_rule None --batch_size 256 --augmentation Geometry --epochs 150 --arch iternet_classifier_data --unfreeze --resume exp/2020-08-08_12_03_05_586405_lr_0_001_Focal_Resample_Geometry/Focal_Resample_epoch_80.pth -= useless

CUDA_VISIBLE_DEVICES=9,10,11,12,13,14,15 python main.py --gpus 9,10,11,12,13,14,15 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 200 --arch EfficientNet =: can be usefull (no data)

CUDA_VISIBLE_DEVICES=9,10,11,12,13,14,15 python main.py --gpus 9,10,11,12,13,14,15 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 200 --arch EfficientNet =: can be usefull (with data)

CUDA_VISIBLE_DEVICES=9,10,11,12,13,14,15 python main.py --gpus 9,10,11,12,13,14,15 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 200 --arch EfficientNet --efficient_depth b4 := can be usefull (with data)

CUDA_VISIBLE_DEVICES=9,10,11,12,13,14,15 python main.py --gpus 9,10,11,12,13,14,15 --train_rule Resample --batch_size 256 --augmentation GeometryContrast --epochs 200 --arch EfficientNet --efficient_depth b4 :=