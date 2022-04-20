> NASBench101 does not support weight-sharing training

1. install `nasbench-101`

```
git clone https://github.com/google-research/nasbench
cd nasbench
pip install -e .
```

2. downlowd tfrecord

```
curl -O https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
curl -O https://storage.googleapis.com/nasbench/nasbench_only_108.tfrecord
```

- nasbench_full.tfrecord includes the results on epoch of 4, 12, 36, 108
- nasbench_only_108.tfrecord only includes the result on epoch of 108

3. convert tfrecord to database

```
python db_gen.py --inputFile /path/to/nasbench_full.tfrecord
```