CUDA_VISIBLE_DEVICES=2,4,6 \
python main_sar.py \
--test_code 0 \
--config configs/_sarResNet50_g4a2s2_ls_warmup.py \
--temp_scheduler linear \
--train_url ./log/test/ \
--no_train_on_cloud \
--use_amp 0 \
--dist_url tcp://127.0.0.1:29699