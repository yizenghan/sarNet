CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_sar_nomask.py \
--test_code 0 \
--config configs/_uniform_res50_g1a2_ls_warmup.py \
--train_url ./log/test2/ \
--no_train_on_cloud \
--use_amp 0 \
--dist_url tcp://127.0.0.1:29679