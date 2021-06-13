import hydra
from omegaconf import OmegaConf

if __name__ == '__main__':
    cfg = OmegaConf.load('../../configs/model/random_nas_model.yaml')
    nas_net = hydra.utils.instantiate(cfg, _recursive_=False)
