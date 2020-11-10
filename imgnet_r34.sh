CUDA_VISIBLE_DEVICES=0,1,2,3 python main_sar_imgnet_optim_rate_local.py \
--config configs/__sar34_g2a2s2m72_ls_warmup_local.py \
--data_url /home/ImageNet/ \
--train_url log/ \
--arch sar_resnet_imgnet4stage_alphaBase --arch_config sar_resnet34_alphaBase_4stage_imgnet \
--patch_groups 4 --alpha 2 --base_scale 2 --beta 1 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 110 --temp_scheduler exp \
--target_rate 1 --lambda_act 1 \
--ta_begin_epoch 0 --ta_last_epoch 110 --dynamic_rate 1 \
--dist_url tcp://127.0.0.1:29503
