exp=$1
name=$2
others=$3

# mpirun -np 2 python run.py \
# mpirun -np 2 python run.py \
# python -m ipdb run.py \
python run.py \
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs] \
experiment=$exp.yaml \
logger.wandb.name=$name \
hydra.job.name=$name \
$others