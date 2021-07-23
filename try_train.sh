CUDA_VISIBLE_DEVICES=0,1 \
python main_sar_local.py \
--test_code 0 \
--config configs/__freFuse50_g1a1s2m71_ls_warmup.py \
--train_url ./log/test/ \
--no_train_on_cloud \
--use_amp 0 \
--target_rate 1.0 \
--lambda_act 0.0 \
--t0 5.0 \
--dynamic_rate 0 \
--optimize_rate_begin_epoch 55 \
--temp_scheduler cosine \
--dist_url tcp://127.0.0.1:29619