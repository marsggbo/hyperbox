> NASBench101 does not support weight-sharing training

1. install `nasbench-101`

- for tensorflow 1.x
```bash
pip install git+https://github.com/google-research/nasbench.git
```

- for tensorflow 2.x
```bash
pip install git+https://github.com/gabikadlecova/nasbench.git
pip install protobuf==3.20.0
```

2. downlowd tfrecord

```
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
wget https://storage.googleapis.com/nasbench/nasbench_only_108.tfrecord
```

- nasbench_full.tfrecord includes the results on epoch of 4, 12, 36, 108
- nasbench_only_108.tfrecord only includes the result on epoch of 108

3. convert tfrecord to database

```
python db_gen.py --inputFile /path/to/nasbench_full.tfrecord
```

# Thanks
- https://github.com/romulus0914/NASBench-PyTorch/blob/master/main.py
- https://nni.readthedocs.io/en/v1.7/NAS/BenchmarksExample.html
- https://github.com/microsoft/nni/blob/v1.7/src/sdk/pynni/nni/nas/benchmarks/nasbench101/__init__.py