CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_sar.py \
--test_code 1 \
--config configs/_sarResNet50_g4_blConfig_ls.py \
--resume evaluate_models/r50_g4_ta05_ls_110ep.pth.tar \
--evaluate \
--train_url ./log/test/ \
--no_train_on_cloud \
--use_amp 0 \
--dist_url tcp://127.0.0.1:29699