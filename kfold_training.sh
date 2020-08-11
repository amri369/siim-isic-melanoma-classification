----------EfficientNet

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python main.py --gpus 1,2,3,4,5,6 --train_rule Resample --batch_size 256 --augmentation GeometryContrast --epochs 200 --arch efficientnet-b7 --train_csv data/splits/train-K-1.csv --val_csv data/splits/test-K-1.csv --model_dir exp/efficientnet-b7-K-1 : Done -> best epoch 121

CUDA_VISIBLE_DEVICES=7,8,9,11,12,13,14 python main.py --gpus 7,8,9,11,12,13,14 --train_rule Resample --batch_size 256 --augmentation GeometryContrast --epochs 200 --arch efficientnet-b7 --train_csv data/splits/train-K-2.csv --val_csv data/splits/test-K-2.csv --model_dir exp/efficientnet-b7-K-2 : Done -> best epoch 147

CUDA_VISIBLE_DEVICES=8,9,11,12,13,14,15 python main.py --gpus 8,9,11,12,13,14,15 --train_rule Resample --batch_size 256 --augmentation GeometryContrast --epochs 200 --arch efficientnet-b7 --train_csv data/splits/train-K-3.csv --val_csv data/splits/test-K-3.csv --model_dir exp/efficientnet-b7-K-3 -> best epoch 145

-----------residual_attention
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --gpus 0,1,2,3,4 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 200 --arch residual_attention --train_csv data/splits/train-K-1.csv --val_csv data/splits/test-K-1.csv --model_dir exp/residual_attention-K-1 --image_dir data/resized-jpeg-448/train/ -> best epoch 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --gpus 0,1,2,3,4 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 200 --arch residual_attention --train_csv data/splits/train-K-2.csv --val_csv data/splits/test-K-2.csv --model_dir exp/residual_attention-K-2 --image_dir data/resized-jpeg-448/train/ -> best epoch 3

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --gpus 0,1,2,3,4 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 200 --arch residual_attention --train_csv data/splits/train-K-3.csv --val_csv data/splits/test-K-3.csv --model_dir exp/residual_attention-K-3 --image_dir data/resized-jpeg-448/train/ -> best epoch 6


----------iternet
CUDA_VISIBLE_DEVICES=7,8,9,11,12,13,14 python main.py --gpus 7,8,9,11,12,13,14 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 200 --arch iternet_extractor_classifier_data --train_csv data/splits/train-K-1.csv --val_csv data/splits/test-K-1.csv --model_dir exp/iternet-K-1 --unfreeze -> best epoch 82

CUDA_VISIBLE_DEVICES=7,8,9,11,12,13,14 python main.py --gpus 7,8,9,11,12,13,14 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 200 --arch iternet_extractor_classifier_data --train_csv data/splits/train-K-2.csv --val_csv data/splits/test-K-2.csv --model_dir exp/iternet-K-2 --unfreeze -> best epoch 78

CUDA_VISIBLE_DEVICES=7,8,9,11,12,13,14 python main.py --gpus 7,8,9,11,12,13,14 --train_rule Resample --batch_size 256 --augmentation Geometry --epochs 200 --arch iternet_extractor_classifier_data --train_csv data/splits/train-K-3.csv --val_csv data/splits/test-K-3.csv --model_dir exp/iternet-K-3 --unfreeze -> best epoch 73


----------EfficientNet pretrained on ISIC2019

CUDA_VISIBLE_DEVICES=1,2,3,4,6 python main.py --gpus 1,2,3,4,6 --train_rule Resample --batch_size 256 --augmentation GeometryContrast --epochs 200 --arch efficientnet-b7 --train_csv data/splits/train-K-1.csv --val_csv data/splits/test-K-1.csv --model_dir exp/efficientnet-b7-K-1 --pretrained

CUDA_VISIBLE_DEVICES=7,8,9,11,12,13 python main.py --gpus 7,8,9,11,12,13 --train_rule Resample --batch_size 256 --augmentation GeometryContrast --epochs 200 --arch efficientnet-b7 --train_csv data/splits/train-K-2.csv --val_csv data/splits/test-K-2.csv --model_dir exp/efficientnet-b7-K-2 --pretrained

----------EfficientNet pretrained on ISIC2019 with new optimizer

CUDA_VISIBLE_DEVICES=1,2,3,4,6 python main.py --gpus 1,2,3,4,6 --train_rule Resample --batch_size 256 --augmentation GeometryContrast --epochs 200 --arch efficientnet-b7 --train_csv data/splits/train-K-1.csv --val_csv data/splits/test-K-1.csv --model_dir exp/efficientnet-b7-K-1 --pretrained --optimizer RMS --scheduler StepLR --lr 0.01 --weight_decay 1E-5

