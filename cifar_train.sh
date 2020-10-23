CUDA_VISIBLE_DEVICES=1 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 1 --alpha 1 --base_scale 2 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 2 --alpha 1 --base_scale 2 &\

CUDA_VISIBLE_DEVICES=2 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 4 --alpha 1 --base_scale 2;

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 1 --alpha 2 --base_scale 2 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 2 --alpha 2 --base_scale 2 &\

CUDA_VISIBLE_DEVICES=2 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 4 --alpha 2 --base_scale 2;

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 1 --alpha 1 --base_scale 4 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 2 --alpha 1 --base_scale 4 &\

CUDA_VISIBLE_DEVICES=2 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 4 --alpha 1 --base_scale 4;

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 1 --alpha 2 --base_scale 4 &\

CUDA_VISIBLE_DEVICES=1 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 2 --alpha 2 --base_scale 4 &\

CUDA_VISIBLE_DEVICES=2 python main_sar_cifar.py \
--config configs/_sarResNet50_g1a1.py \
--patch_groups 4 --alpha 2 --base_scale 4;
