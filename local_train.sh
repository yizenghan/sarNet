CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_sar_local2.py \
--test_code 0 \
--config configs/_baseAlpha50_g4a2s2_ls_warmup_local.py \
--train_url ./log/_baseAlpha_g4a2s2_ls_warmup_nota2/ \
--no_train_on_cloud \
--use_amp 0 \
--target_rate 1.0 \
--lambda_act 0.0 \
--t0 5.0 \
--dynamic_rate 0 \
--optimize_rate_begin_epoch 55 \
--temp_scheduler cosine \
--dist_url tcp://127.0.0.1:29676