CUDA_VISIBLE_DEVICES=2 python main_sar_cifar_optim_rate.py \
--data 'cifar100' \
--arch 'sar_resnet_cifar4stage_alphaBase' --arch_config 'sar_resnet50_alphaBase_4stage_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.25 --lambda_act 1.0 \
--ta_begin_epoch 150 --ta_last_epoch 300 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=3 python main_sar_cifar_optim_rate.py \
--data 'cifar100' \
--arch 'sar_resnet_cifar4stage_alphaBase' --arch_config 'sar_resnet50_alphaBase_4stage_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.4 --lambda_act 1.0 \
--ta_begin_epoch 150 --ta_last_epoch 300 --dynamic_rate 0 \
--round 1 ;

CUDA_VISIBLE_DEVICES=2 python main_sar_cifar_optim_rate.py \
--data 'cifar100' \
--arch 'sar_resnet_cifar4stage_alphaBase' --arch_config 'sar_resnet50_alphaBase_4stage_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.7 --lambda_act 1.0 \
--ta_begin_epoch 150 --ta_last_epoch 300 --dynamic_rate 0 \
--round 1 &\

CUDA_VISIBLE_DEVICES=3 python main_sar_cifar_optim_rate.py \
--data 'cifar100' \
--arch 'sar_resnet_cifar4stage_alphaBase' --arch_config 'sar_resnet50_alphaBase_4stage_cifar' \
--patch_groups 2 --alpha 2 --base_scale 2 --beta 1 --width 2 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 1.0 --lambda_act 1.0 \
--ta_begin_epoch 150 --ta_last_epoch 300 --dynamic_rate 1 \
--round 1 &\