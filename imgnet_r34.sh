# CUDA_VISIBLE_DEVICES=0,1,2,3 python main_sar_imgnet_optim_rate_local_nowait.py \
# --config 'configs/__sar34_g2a2s2m72_256ls_warmup_local.py' \
# --workers 32 \
# --train_url './log/___12/' \
# --data_url /data/ImageNet/ \
# --resume log/_ImageNet/sar_resnet34_alphaBase_4stage_imgnet_OptimRate/g2_a2b1_s2/t0_1_0_tLast0_01_tempScheduler_exp_target0_75_optimizeFromEpoch30to60_dr2_lambda_1_0/checkpoint.pth.tar \
# --arch 'sar_resnet_imgnet4stage_alphaBase' --arch_config 'sar_resnet34_alphaBase_4stage_imgnet' \
# --patch_groups 2 --alpha 2 --base_scale 2 --beta 1 \
# --t0 1.0 --t_last 1e-2 --t_last_epoch 110 --temp_scheduler 'exp' \
# --target_rate 0.75 --lambda_act 1.0 \
# --ta_begin_epoch 30 --ta_last_epoch 60 --dynamic_rate 2 \
# --dist_url 'tcp://127.0.0.1:29501' 

CUDA_VISIBLE_DEVICES=4,5,6,7 python main_sar_imgnet_optim_rate_local_nowait.py \
--config 'configs/__sar34_g2a2s2m72_256ls_warmup_local.py' \
--workers 32 \
--train_url './log/___12/' \
--data_url /data/ImageNet/ \
--arch 'sar_resnet_imgnet4stage_alphaBase' --arch_config 'sar_resnet34_alphaBase_4stage_imgnet' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 110 --temp_scheduler 'exp' \
--target_rate 0.5 --lambda_act 1.0 \
--ta_begin_epoch 30 --ta_last_epoch 60 --dynamic_rate 2 \
--dist_url 'tcp://127.0.0.1:29502' 