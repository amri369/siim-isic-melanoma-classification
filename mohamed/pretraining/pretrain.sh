-----------residual_attention
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --gpus 0,1,2,3,4 --train_rule None --lr 0.01 --batch_size 256 --augmentation Geometry --epochs 200 --arch residual_attention --train_csv data/ISIC2019/train_split.csv --val_csv data/ISIC2019/val_split.csv --model_dir exp/residual-attention-pretraining --image_dir data/ISIC2019/ISIC_2019_Training_Input/ --num_classes 8 --weight_decay 1E-3

-----------efficient_net-b7
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --gpus 0,1,2,3,4 --train_rule None --batch_size 256 --augmentation Geometry --epochs 200 --arch efficientnet-b7 --train_csv data/ISIC2019/train_split.csv --val_csv data/ISIC2019/val_split.csv --model_dir exp/efficientnet-b7-pretraining --image_dir data/ISIC2019/ISIC_2019_Training_Input/ --num_classes 8


-----------efficient_net-b7 Higher resolution
CUDA_VISIBLE_DEVICES=0,7,8,9,11,12,13,14 python main.py --gpus 0,7,8,9,11,12,13,14 --train_rule None --batch_size 256 --augmentation Geometry --epochs 200 --arch efficientnet-b7 --train_csv data/ISIC2019/train_split.csv --val_csv data/ISIC2019/val_split.csv --model_dir exp/efficientnet-b7-448-pretraining --image_dir data/ISIC2019/ISIC_2019_Training_Input/ --num_classes 8


-----------iternet
CUDA_VISIBLE_DEVICES=7,8,9,11,12,13,14 python main.py --gpus 7,8,9,11,12,13,14 --train_rule None --lr 0.01 --batch_size 256 --augmentation Geometry --epochs 200 --arch iternet_extractor_classifier_data --train_csv data/ISIC2019/train_split.csv --val_csv data/ISIC2019/val_split.csv --model_dir exp/iternet-pretraining --image_dir data/ISIC2019/ISIC_2019_Training_Input/ --num_classes 8