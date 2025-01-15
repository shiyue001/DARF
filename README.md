# GARF-Advanced version
including 4 options: 
(1) the version of the draft;
   
new attempts:
(2) use optical flow-based feature extraction;
(3) attempt to use FFC instead of the ResNet;
(4) use attention to fuse the color at the decoder.

train:
python train.py --config configs/train_dtu.txt --ckpt_path ../code/IBR_vsr/results/dtu_depth0.5_5views/model_045000.pth --num_source_views 5 --expname dtu_depth0.5_5views

test:
cd eval
python eval.py --config ../configs/eval_llff.txt --ckpt_path ../experiments/IBR/IBRNet_ffc/model_020000.pth --use_ffc --num_source_views
