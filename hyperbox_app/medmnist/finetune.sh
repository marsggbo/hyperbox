datamodule=$1
others=$2

CUDA_VISIBLE_DEVICES=3 python run.py \
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] \
experiment=finetune.yaml \
datamodule=$datamodule \
datamodule.batch_size=128 \
logger.wandb.name=medmnist_finetune_$datamodule \
hydra.job.name=medmnist_finetune_$datamodule \
$others

