CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_sar.py \
--test_code 1 \
--config configs/_sarResNet50_g2a2_ls_warmup.py \
--resume ../sar_evaluate_models/model_best.pth.tar \
--evaluate \
--train_url ./log/test/ \
--no_train_on_cloud \
--use_amp 0 \
--dist_url tcp://127.0.0.1:29699