CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_sar_nomask.py \
--test_code 0 \
--config configs/_sarResNet50_mask1_g1_blConfig_ls.py \
--train_url ./log/test/ \
--no_train_on_cloud \
--use_amp 0 \
--dist_url tcp://127.0.0.1:29699