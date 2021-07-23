CUDA_VISIBLE_DEVICES=2 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 160 --temp_scheduler 'exp' \
--target_rate 1.0 --lambda_act 1.0 \
--ta_begin_epoch 160 --ta_last_epoch 160 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=2 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 160 --temp_scheduler 'exp' \
--target_rate 0.6 --lambda_act 1.0 \
--ta_begin_epoch 80 --ta_last_epoch 160 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=2 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 160 --temp_scheduler 'exp' \
--target_rate 0.4 --lambda_act 1.0 \
--ta_begin_epoch 80 --ta_last_epoch 160 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=2 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 160 --temp_scheduler 'exp' \
--target_rate 0.25 --lambda_act 1.0 \
--ta_begin_epoch 80 --ta_last_epoch 160 --dynamic_rate 0 \
--round 1 &\


CUDA_VISIBLE_DEVICES=3 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 160 --temp_scheduler 'exp' \
--target_rate 0.6 --lambda_act 1.0 \
--ta_begin_epoch 60 --ta_last_epoch 100 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=3 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 160 --temp_scheduler 'exp' \
--target_rate 0.4 --lambda_act 1.0 \
--ta_begin_epoch 60 --ta_last_epoch 100 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=3 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 160 --temp_scheduler 'exp' \
--target_rate 0.25 --lambda_act 1.0 \
--ta_begin_epoch 60 --ta_last_epoch 100 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=3 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 160 --temp_scheduler 'exp' \
--target_rate 0.25 --lambda_act 1.0 \
--ta_begin_epoch 80 --ta_last_epoch 120 --dynamic_rate 2 \
--round 1 &\