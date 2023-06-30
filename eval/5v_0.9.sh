#!/bin/bash

#SBATCH --tmp=4000
#SBATCH --job-name=5_0.8
#SBATCH --error=5_0.8.err

unset PYTHONPATH 
source /cluster/project/cvl/yueshi/env/ibr1/bin/activate
python eval.py --config ../configs/eval_dtu.txt --ckpt_path /cluster/project/cvl/yueshi/code/IBR_vsr/results/dtu_depth0.9_5views/model_250000.pth --num_source_views 5 --expname dtu_5v_0.9