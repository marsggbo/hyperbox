num=$1
BS=32
exp=example_ofa_nas
if [ $num -eq 1 ]
then
{  
    echo "python run.py experiment=$exp.yaml logger.wandb.name=g${num}_bs${BS} trainer.gpus=${num} datamodule.batch_size=${BS}"
    python run.py experiment=$exp.yaml logger.wandb.name=g${num}_bs${BS} trainer.gpus=${num} datamodule.batch_size=${BS}
}
elif [ $num -le 4 ]
then
{
    echo "python run.py experiment=$exp.yaml logger.wandb.name=ddp_g${num}_bs${BS} trainer.gpus=${num} trainer.accelerator=ddp datamodule.batch_size=${BS}"
    python run.py experiment=$exp.yaml logger.wandb.name=ddp_g${num}_bs${BS} trainer.gpus=${num} trainer.accelerator=ddp datamodule.batch_size=${BS}
}
else
{
    echo "#gpus should less than 5"    
}
# mpirun -np ${num} python run.py experiment=$exp.yaml logger.wandb.name=ddp_g${num} trainer.gpus=1 trainer.accelerator=horovod
fi