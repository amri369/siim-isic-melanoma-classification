CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 python ensemble.py --validate --use_gpu --ensemble_method rankdata

CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 python ensemble.py --image_dir data/resized-jpeg-448/test/ --csv_path data/test.csv --use_gpu --ensemble_method rankdata

# min -> 0.952988169721
# average -> 0.952988169721