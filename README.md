# GARF-Advanced version
including 4 options: 1) the version of the paper; 2) use optical flow-based feature extraction; 3) attempt to use FFC instead of the ResNet; 4) use attention to fuse the color at the decoder.

train:

python train.py --config configs/pretrain.txt 

python train.py --config configs/pretrain.txt --use_ffc --num_source_views --expname

python train.py --config configs/pretrain.txt --use_attn --num_source_views --expname 

test:

cd eval

python eval.py --config ../configs/eval_llff.txt --ckpt_path ../experiments/IBR/IBRNet_ffc/model_020000.pth --use_ffc --num_source_views
