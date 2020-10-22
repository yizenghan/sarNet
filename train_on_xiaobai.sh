CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_sar_local2.py \
--test_code 0 \
--config configs/__r34_ls_warmup.py \
--train_url ./log/__r34_baseline/ \
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