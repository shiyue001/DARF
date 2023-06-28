# IBR_VSR
including 4 options: 1) add depth prior to the sampling of IBR; 2) use optical flow based feature extraction; 3) use ffc instead of resnet; 4) use attentio to fuse the color at decoder

env: ibr1

train:
python train.py --config configs/pretrain.txt 

python train.py --config configs/pretrain.txt --use_ffc --num_source_views --expname

python train.py --config configs/pretrain.txt --use_attn --num_source_views --expname 

test:
cd eval
python eval.py --config ../configs/eval_llff.txt --ckpt_path /cluster/work/cvl/yueshi/experiments/IBR/IBRNet_ffc/model_020000.pth --use_ffc --num_source_views
