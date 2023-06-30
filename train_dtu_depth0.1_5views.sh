#!/bin/bash

#SBATCH --tmp=4000
#SBATCH --job-name=0.1depth
#SBATCH --error=0.1depth.err

unset PYTHONPATH 
source /cluster/project/cvl/yueshi/env/ibr1/bin/activate
python train.py --config configs/train_dtu.txt --ckpt_path /cluster/project/cvl/yueshi/code/IBR_vsr/results/dtu_depth0.1_5views/model_045000.pth --num_source_views 5 --expname dtu_depth0.1_5views
