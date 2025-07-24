export CUDA_VISIBLE_DEVICES=3 && python main_eval.py \
  --dataset_file nightclear \  #domain_name
  --output_dir logs/test \
	-c config/DINO_4scale_test.py   \
	--eval --resume checkpoint_best_regular.pth \
	--options dn_scalar=100n embed_iit_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
