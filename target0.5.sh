CUDA_VISIBLE_DEVICES=3 python main_sar_cifar_optim_rate.py \
--dataset 'cifar100' \
--arch 'sar_resnet_cifar4stage_alphaBase' --arch_config 'sar_resnet50_alphaBase_4stage_cifar' \
--patch_groups 4 --alpha 2 --base_scale 2 --beta 1 \
--epochs 300 --lr 0.2 --batch_size 128 \
--t0 1.0 --t_last 1e-2 --t_last_epoch 300 --temp_scheduler 'exp' \
--target_rate 0.5 --lambda_act 1.0 \
--ta_begin_epoch 90 --ta_last_epoch 150 --dynamic_rate 2 \
--round 2