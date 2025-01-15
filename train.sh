#!/bin/bash

#SBATCH --tmp=4000
#SBATCH --job-name=0.1depth
#SBATCH --error=0.1depth.err

unset PYTHONPATH 
source /cluster/project/yue/env/ibr1/bin/activate
python train.py --config configs/train_dtu.txt --ckpt_path ../code/IBR_vsr/results/dtu_depth0.5_5views/model_045000.pth --num_source_views 5 --expname dtu_depth0.5_5views
