CUDA_VISIBLE_DEVICES=2 python run.py \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox/hyperbox_app/multiple_path_nas/configs] \
experiment=multipath_201_darts.yaml \

# CUDA_VISIBLE_DEVICES=0 python run.py hydra.searchpath=[file:///home/xihe/xinhe/hyperbox/hyperbox_app/multiple_path_nas/configs] experiment=

# python -m ipdb run.py \
# hydra.searchpath=[file:///home/xihe/xinhe/hyperbox/hyperbox_app/multiple_path_nas/configs] \
# experiment=multiple_path_nas.yaml \
# ++model.supernet_epoch=0 \