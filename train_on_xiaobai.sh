
# python3 opt_gs_xiaobai.py \
# --data imagenet \
# --data_url /home/CLS-LOC/ \
# --visible_gpus 0,1,2,3 \
# --arch patch_group_resnext \
# --arch_config resnext50_32x4d \
# --batch_size 256 \
# --workers 64 \
# --lr 0.1 \
# --weight_decay 1e-4 \
# --epochs 90 \
# --train_url ./log/resnext50_32x4d_patchgroups4_target03_from0/ \
# --print_freq 200 \
# --schedule_iter \
# --lambda_act 1 --t0 0.5 --target_rate 0.3 --optimize_rate_begin_epoch 0 --patch_groups 4 \
# --dist-url 'tcp://127.0.0.1:11691' \



# python3 opt_gs_xiaobai.py \
# --data imagenet \
# --data_url /home/CLS-LOC/ \
# --visible_gpus 4,5,6,7 \
# --arch patch_refine_mobilenet_v2 \
# --arch_config mobilenet_v2 \
# --resume log/patch_refine_mobilenet_v2_lamda10_ta02_start45_g8_addConv/checkpoint.pth.tar \
# --batch_size 256 \
# --workers 64 \
# --lr 0.05 \
# --weight_decay 4e-5 \
# --epochs 150 \
# --train_url ./log/patch_refine_mobilenet_v2_lamda10_ta02_start45_g8_addConv_resume/ \
# --print_freq 200 \
# --schedule_iter \
# --lambda_act 1.0 --t0 0.1 --target_rate 0.2 --optimize_rate_begin_epoch 45 --patch_groups 8 \
# --dist-url 'tcp://127.0.0.1:11692' \

# python3 opt_gs_xiaobai.py \
# --data imagenet \
# --data_url /home/CLS-LOC/ \
# --visible_gpus 0,1,2,3 \
# --arch patch_refine_mobilenet_v2 \
# --arch_config mobilenet_v2 \
# --batch_size 256 \
# --workers 64 \
# --lr 0.05 \
# --weight_decay 4e-5 \
# --epochs 150 \
# --train_url ./log/patch_refine_mobilenet_v2_lamda05_t03_ta02_start75_g8_addBN_150eps/ \
# --print_freq 400 \
# --schedule_iter \
# --lambda_act 0.5 --t0 0.3 --target_rate 0.2 --optimize_rate_begin_epoch 75 --patch_groups 8 \
# --dist-url 'tcp://127.0.0.1:11693' \

python3 opt_gs_xiaobai.py \
--data imagenet \
--data_url /home/CLS-LOC/ \
--visible_gpus 0,1,2,3 \
--arch base_refineGroup_mobilenet_v2 \
--arch_config mobilenet_v2 \
--batch_size 256 \
--workers 64 \
--lr 0.05 \
--weight_decay 4e-5 \
--epochs 150 \
--train_url ./log/g2_lambda02from0_ta03/ \
--print_freq 400 \
--schedule_iter \
--lambda_act 0.2 --t0 0.5 --target_rate 0.3 --optimize_rate_begin_epoch 0 --patch_groups 2 \
--dist-url 'tcp://127.0.0.1:11661' \



# python3 imagenet_xiaobai.py \
# --data imagenet \
# --data_url /home/CLS-LOC/ \
# --visible_gpus 4,5,6,7 \
# --arch mobilenet_v2 \
# --arch_config mobilenet_v2_width10 \
# --input_size 192 \
# --batch_size 256 \
# --workers 64 \
# --lr 0.05 \
# --weight_decay 4e-5 \
# --epochs 150 \
# --train_url ./log/mobilenet_width10_input192/ \
# --print_freq 400 \
# --schedule_iter \
# --dist-url 'tcp://127.0.0.1:11700' \