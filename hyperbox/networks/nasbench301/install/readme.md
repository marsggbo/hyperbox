## NASBench301 Installation Guidence

## Environments

ubuntu16.04
cuda11.1
torch=1.8+cu111
python=3.7.0

如果是cuda 10.2,执行以下命令：

```
cat requirements_cuda102.txt | xargs -n 1 -L 1 pip install
```

如果是cuda11.1，执行以下命令：
```
cat requirements_cuda111.txt | xargs -n 1 -L 1 pip install
```

## Install NB301

```
$ git clone https://github.com/automl/nasbench301
$ cd nasbench301
$ pip install .
```

## Download weights

v0.9: [nasbench301_models_v0.9_zip](https://figshare.com/articles/software/nasbench301_models_v0_9_zip/12962432)

v1.0: [nasbench301_models_v1_0_zip](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510)

运行官方demo:

```
cd nasbench301/nasbench301
unzip nasbench301_models_v1.0.zip
mv nb_models nb_models_1.0
python example.py
```

运行结果：

```
==> Loading performance surrogate model...
/home/pdluser/project/nasbench301/nasbench301/nb_models_1.0/xgb_v1.0
==> Loading runtime surrogate model...
==> Creating test configs...
==> Predict runtime and performance...
Genotype architecture performance: 94.167947, runtime 4781.153014
Configspace architecture performance: 91.834275, runtime 4285.378243
```

## Using NB301 in Hyperbox

replace `default_path` to your download path `/path/to/your/downloads/nb_models_1.0`