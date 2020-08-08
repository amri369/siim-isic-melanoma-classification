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

python main.py --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 400 --lr 0.001 --arch iternet_data --train_csv data/train_split.csv --resume exp/2020-08-07_05_48_12_199703_lr_0_001_Focal_Power_GeometryContrast/Focal_Power_epoch_299.pth

