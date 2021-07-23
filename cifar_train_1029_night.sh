CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 160 --temp_scheduler 'cosine' \
--target_rate 1.0 --lambda_act 0.5 \
--optimize_rate_begin_epoch 161 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 160 --temp_scheduler 'cosine' \
--target_rate 0.6 --lambda_act 0.5 \
--optimize_rate_begin_epoch 80 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 160 --temp_scheduler 'cosine' \
--target_rate 0.4 --lambda_act 0.5 \
--optimize_rate_begin_epoch 80 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 160 --temp_scheduler 'cosine' \
--target_rate 0.25 --lambda_act 0.5 \
--optimize_rate_begin_epoch 80 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 1.0 --lambda_act 0.5 \
--optimize_rate_begin_epoch 161 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 0.6 --lambda_act 0.5 \
--optimize_rate_begin_epoch 45 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 0.4 --lambda_act 0.5 \
--optimize_rate_begin_epoch 45 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 0.25 --lambda_act 0.5 \
--optimize_rate_begin_epoch 45 --dynamic_rate 0 \
--round 1;

CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 1.0 --lambda_act 0.5 \
--optimize_rate_begin_epoch 161 --dynamic_rate 1 \
--round 1 &\

CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 0.6 --lambda_act 0.5 \
--optimize_rate_begin_epoch 45 --dynamic_rate 1 \
--round 1 &\

CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 0.4 --lambda_act 0.5 \
--optimize_rate_begin_epoch 45 --dynamic_rate 1 \
--round 1 &\

CUDA_VISIBLE_DEVICES=0 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 0.25 --lambda_act 0.5 \
--optimize_rate_begin_epoch 45 --dynamic_rate 1 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 1.0 --lambda_act 0.5 \
--optimize_rate_begin_epoch 161 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 0.6 --lambda_act 0.5 \
--optimize_rate_begin_epoch 45 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 0.4 --lambda_act 0.5 \
--optimize_rate_begin_epoch 45 --dynamic_rate 2 \
--round 1 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar_optim_rate.py \
--arch 'sar_resnet_cifar3stage_alphaBase' --arch_config 'sar_resnet32x2_alphaBase_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 160 --lr 0.1 --batch_size 64 \
--t0 3.0 --t_last 1e-6 --t_last_epoch 90 --temp_scheduler 'cosine' \
--target_rate 0.25 --lambda_act 0.5 \
--optimize_rate_begin_epoch 45 --dynamic_rate 2 \
--round 1 &\