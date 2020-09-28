CUDA_VISIBLE_DEVICES=7,0,1,2 \
python main_sar.py \
--test_code 0 \
--config configs/sar_BLConfig.py \
--train_url ./log/test/ \
--no_train_on_cloud \
--dist_url tcp://127.0.0.1:29699