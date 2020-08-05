python main.py --resume exp/iternet/Focal_Resample_epoch_199.pth --train_rule Reweight --epochs 300
python main.py --resume exp/iternet/Focal_Resample_epoch_171.pth --train_rule Reweight --epochs 300
python main.py --train_rule Resample --batch_size 32 --augmentation ImageNetPolicy --epochs 200