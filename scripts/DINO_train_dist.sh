export CUDA_VISIBLE_DEVICES=2,3 && TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
	--output_dir logs/train  \
	-c config/DINO_4scale.py   \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0