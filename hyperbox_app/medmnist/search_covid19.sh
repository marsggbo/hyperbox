exp=$1 
# gdas_ccccii
# gdas_nii
# gdas_iran
others=$2

CUDA_VISIBLE_DEVICES=1 python run.py \
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] \
experiment=$exp \
datamodule.img_size=[160,160] \
datamodule.center_size=[120,120] \
datamodule.slice_num=32 \
datamodule.num_workers=3 \
datamodule.batch_size=64 \
model/optimizer_cfg=sgd \
model.optimizer_cfg.lr=0.025 \
model.metric_cfg._target_=hyperbox.utils.metrics.Accuracy \
trainer.gpus=1 \
trainer.accelerator=dp \
$others

# experiment=gdas_ccccii.yaml \
# CUDA_VISIBLE_DEVICES=0 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] experiment=gdas_nii.yaml

# CUDA_VISIBLE_DEVICES=2 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] experiment=gdas_iran.yaml