# CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
# --arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
# --patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
# --epochs 300 --lr 0.2 --batch_size 128 \
# --t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
# --target_rate 1.0 --lambda_act 1.0 \
# --ta_begin_epoch 300 --ta_last_epoch 300 --dynamic_rate 0 \
# --round 1 &\

# CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
# --arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
# --patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
# --epochs 300 --lr 0.2 --batch_size 128 \
# --t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
# --target_rate 0.6 --lambda_act 1.0 \
# --ta_begin_epoch 150 --ta_last_epoch 300 --dynamic_rate 0 \
# --round 1 &\

# CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
# --arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
# --patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
# --epochs 300 --lr 0.2 --batch_size 128 \
# --t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
# --target_rate 0.4 --lambda_act 1.0 \
# --ta_begin_epoch 150 --ta_last_epoch 300 --dynamic_rate 0 \
# --round 1 &\

# CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
# --arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
# --patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
# --epochs 300 --lr 0.2 --batch_size 128 \
# --t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
# --target_rate 0.25 --lambda_act 1.0 \
# --ta_begin_epoch 150 --ta_last_epoch 300 --dynamic_rate 0 \
# --round 1 &\


CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.6 --lambda_act 1.0 \
--ta_begin_epoch 90 --ta_last_epoch 150 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.4 --lambda_act 1.0 \
--ta_begin_epoch 90 --ta_last_epoch 150 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.25 --lambda_act 1.0 \
--ta_begin_epoch 90 --ta_last_epoch 150 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.6 --lambda_act 1.0 \
--ta_begin_epoch 150 --ta_last_epoch 210 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.4 --lambda_act 1.0 \
--ta_begin_epoch 150 --ta_last_epoch 210 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.25 --lambda_act 1.0 \
--ta_begin_epoch 150 --ta_last_epoch 210 --dynamic_rate 2 \
--round 1 &\