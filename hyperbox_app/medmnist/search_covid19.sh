CUDA_VISIBLE_DEVICES=1,2,3 python run.py \
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] \
experiment=gdas_ccccii.yaml \
datamodule.img_size=[128,128] \
datamodule.center_size=[96,96] \
datamodule.num_workers=3 \
datamodule.batch_size=128 \
model.metric_cfg._target_=hyperbox.utils.metrics.Accuracy \
trainer.gpus=3

# CUDA_VISIBLE_DEVICES=0 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] experiment=gdas_nii.yaml

# CUDA_VISIBLE_DEVICES=2 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] experiment=gdas_iran.yaml