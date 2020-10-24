CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_sar_local.py \
--test_code 0 \
--config configs/__freFuse34_g2a2s2m71_ls_warmup_local.py \
--train_url ./log/__freFuse34_g2a2s2m71_ls_warmup_resume2/ \
--resume log/__freFuse34_g2a2s2m71_ls_warmup_resume/checkpoint.pth.tar \
--no_train_on_cloud \
--workers 128 \
--use_amp 0 \
--target_rate 1.0 \
--lambda_act 0.0 \
--t0 5.0 \
--dynamic_rate 0 \
--optimize_rate_begin_epoch 55 \
--temp_scheduler cosine \
--dist_url tcp://127.0.0.1:29619