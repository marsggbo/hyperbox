# author: pprp
# date: 2021-10-31
# function: default modelv1.0
# install:
'''
git clone https://github.com/automl/nasbench301.git
cd nasbench301
cat requirements.txt | xargs -n 1 -L 1 pip install
pip install .
'''

import os
from collections import namedtuple

import nasbench301 as nb
from ConfigSpace.read_and_write import json as cs_json
from nasbench301.surrogate_models import ensemble

default_version = "1.0"

default_path = "~/.hyperbox/nb_models"

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def load_model():
    model_paths = {
        model_name: os.path.join(default_path, '{}_v1.0'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    print("==> Loading performance surrogate model...")
    ensemble_dir_performance = model_paths['xgb']
    performance_model = nb.load_ensemble(ensemble_dir_performance)

    print("==> Loading runtime surrogate model...")
    ensemble_dir_runtime = model_paths['lgb_runtime']
    runtime_model = nb.load_ensemble(ensemble_dir_runtime)
    return performance_model, runtime_model


def generate_results(genotype_config):
    # pmodel: prediction model
    # rmodel: runtime model
    pmodel, rmodel = load_model()
    print("==> Creating test configs...")
    genotype_config = Genotype(
        normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
        normal_concat=[2, 3, 4, 5],
        reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
                ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
        reduce_concat=[2, 3, 4, 5]
    ) if genotype_config is None else genotype_config

    prediction_genotype = pmodel.predict(
        config=genotype_config,  representation="genotype", with_noise=True)
    runtime_genotype = rmodel.predict(
        config=genotype_config, representation="genotype")

    return prediction_genotype, runtime_genotype


def sample_config():
    configspace_path = './hyperbox/networks/nasbench301/configspace.json'
    with open(configspace_path, "r") as f:
        json_string = f.read()
        configspace = cs_json.read(json_string)
    configspace_config = configspace.sample_configuration()
    return configspace_config


def test1(genotype_config=None):
    p, r = generate_results(genotype_config)
    print("Genotype architecture performance: %f, runtime %f" % (p, r))


def test2():
    # pmodel: prediction model
    # rmodel: runtime model
    pmodel, rmodel = load_model()
    configspace_config = sample_config()
    prediction_configspace = pmodel.predict(
        config=configspace_config, representation="configspace", with_noise=True)
    runtime_configspace = rmodel.predict(
        config=configspace_config, representation="configspace")
    print("Configspace architecture performance: %f, runtime %f" %
          (prediction_configspace, runtime_configspace))


if __name__ == "__main__":
    test1()
    test2()
