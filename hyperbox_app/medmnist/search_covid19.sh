exp=$1 
# gdas_ccccii
# gdas_nii
# gdas_iran
others=$2

CUDA_VISIBLE_DEVICES=0,1 python run.py \
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] \
experiment=$exp \
datamodule.img_size=[512,512] \
datamodule.center_size=[360,360] \
datamodule.slice_num=32 \
datamodule.num_workers=4 \
datamodule.batch_size=16 \
model.metric_cfg._target_=hyperbox.utils.metrics.Accuracy \
trainer.gpus=2 \
$others

# experiment=gdas_ccccii.yaml \
# CUDA_VISIBLE_DEVICES=0 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] experiment=gdas_nii.yaml

# CUDA_VISIBLE_DEVICES=2 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] experiment=gdas_iran.yaml