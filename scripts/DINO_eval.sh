export CUDA_VISIBLE_DEVICES=2 && python main_eval.py \
  --dataset_file dayclear \
  --output_dir logs/test \
	-c config/DINO_4scale.py   \
	--eval --resume /data/jianhonghan/code/第三篇域泛化/code/github提交/checkpoint_best_regular.pth \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
