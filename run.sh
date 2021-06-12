#!/bin/bash

num=$1
remark=$2
others=$3

BS=64
exp=example_ofa_nas


if [ $num -le 1 ]
then
{  
    echo "python run.py experiment=$exp.yaml logger.wandb.name=g${num}_bs${BS}_${remark} trainer.gpus=${num} datamodule.batch_size=${BS} ${others}"
    python run.py \
    experiment=$exp.yaml \
    logger.wandb.name=g${num}_bs${BS}_${remark} \
    trainer.gpus=${num} \
    datamodule.batch_size=${BS} \
    logger.wandb.offline=True \
    $others
}
else
{
    echo "python run.py experiment=$exp.yaml logger.wandb.name=ddp_g${num}_bs${BS}_${remark} trainer.gpus=${num} trainer.accelerator=ddp datamodule.batch_size=${BS} ${others}"
    python run.py \
    experiment=$exp.yaml \
    logger.wandb.name=ddp_g${num}_bs${BS}_${remark} \
    trainer.gpus=${num} \
    trainer.accelerator=ddp \
    datamodule.batch_size=${BS} \
    logger.wandb.offline=True \
    $others
}
# mpirun -np ${num} python run.py experiment=$exp.yaml logger.wandb.name=ddp_g${num} trainer.gpus=1 trainer.accelerator=horovod
fi