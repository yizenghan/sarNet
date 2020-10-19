CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_sar.py \
--test_code 0 \
--config configs/_sarResNet50_refineOnly_g1a1_ls_warmup.py \
--train_url ./log/test2/ \
--no_train_on_cloud \
--use_amp 0 \
--dist_url tcp://127.0.0.1:29679