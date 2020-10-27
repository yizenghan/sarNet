# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main_sar_local.py \
# --test_code 0 \
# --config configs/__freFuse34_g2a2s2m71_ls_warmup_local.py \
# --train_url ./log/__freFuse34_g2a2s2m71_ls_warmup_resume3/ \
# --resume log/__freFuse34_g2a2s2m71_ls_warmup_resume2/checkpoint.pth.tar \
# --no_train_on_cloud \
# --workers 128 \
# --use_amp 0 \
# --target_rate 1.0 \
# --lambda_act 0.0 \
# --t0 5.0 \
# --dynamic_rate 0 \
# --optimize_rate_begin_epoch 55 \
# --temp_scheduler cosine \
# --dist_url tcp://127.0.0.1:29619


# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main_sar_imgnet_optim_flops.py \
# --test_code 0 \
# --config configs/_baseAlpha50_g4a2s2_ls_warmup_local.py \
# --train_url ./log/_optimFLOPs_baseAlpha50_g4a2s2/ \
# --patch_groups 4 --alpha 2 --base_scale 2 \
# --no_train_on_cloud \
# --workers 128 \
# --use_amp 0 \
# --target_flops 2.25 \
# --lambda_act 0.1 \
# --t0 5.0 \
# --dynamic_rate 0 \
# --optimize_rate_begin_epoch 55 \
# --temp_scheduler cosine \
# --dist_url tcp://127.0.0.1:29619


CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_sar_local2.py \
--test_code 0 \
--config configs/_oct_r50_ls_warmup.py \
--train_url ./log/__octave_resnet50/ \
--no_train_on_cloud \
--workers 128 \
--use_amp 0 \
--dist_url tcp://127.0.0.1:29619