# NAS-Bench-ASR

## 1. Download datasets

```bash
if [ -z "${NASBENCHMARK_DIR}" ]; then
    NASBENCHMARK_DIR=~/.hyperbox/nasbenchasr/
fi

echo "Downloading NAS-Bench-ASR..."
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-bench-gtx-1080ti-fp32.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-bench-jetson-nano-fp32.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-e10-1234.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-e40-1234.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-e40-1235.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-e5-1234.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-info.pickle

mkdir -p ${NASBENCHMARK_DIR}
mv nb-asr*.pickle ${NASBENCHMARK_DIR}
```

## 2. Usage

```python
from hyperbox.mutator import RandomMutator
model = NASBenchASR()
print(sum([p.numel() for p in model.parameters()]))
rm = RandomMutator(model)
rm.reset()

B, F, T = 2, 80, 30
for T in [16]:
    x = torch.rand(B, F, T)
    rm.reset()
    y = model(x)
    print(y.shape)
    # print(rm._cache, len(rm._cache))


list_desc = [
    ['linear', 1],
    ['conv5', 1, 0],
    ['conv7d2', 1, 0, 1],
]
mask = NASBenchASR.list_desc_to_dict_mask(list_desc)
# print(mask)
model2 = NASBenchASR(mask=mask)
print(sum([p.numel() for p in model2.parameters()]))
print(model2.arch_size((B, F, T)))
y = model(x)
print(NASBenchASR.dict_mask_to_list_desc(mask))

print(model2.query_full_info())
print(model2.query_flops())
print(model2.query_latency())
print(model2.query_params())
print(model2.query_test_acc())
print(model2.query_val_acc())
```

输出结果（带有随机性）：
```bash
84867649
torch.Size([2, 4, 49])
39733249
(181.789408, 151.5703125)
[['linear', 1], ['conv5', 1, 0], ['conv7d2', 1, 0, 1]]
{'val_per': [0.9687853, 0.87188685, 0.7872086, 0.6666002, 0.5817228, 0.50474864, 0.4540081, 0.4089128, 0.3726506, 0.35265988, 0.33154014, 0.30796307, 0.29800093, 0.28405392, 0.27661553, 0.270439, 0.26074252, 0.2551637, 0.25124526, 0.2481238, 0.24918643, 0.24540082, 0.24041975, 0.2369662, 0.2374311, 0.2337119, 0.2337119, 0.23391114, 0.2327821, 0.22939497, 0.22893007, 0.22879724, 0.22753537, 0.22992627, 0.22985986, 0.22839876, 0.22461313, 0.22381617, 0.22275354, 0.22328486], 'test_per': 0.24767844378948212, 'arch_vec': [(0, 1), (1, 1, 0), (4, 1, 0, 1)], 'model_hash': 'adb47992d93622245376905cc956a149', 'seed': 1236, 'jetson-nano-fp32': {'latency': 0.578345775604248}, 'gtx-1080ti-fp32': {'latency': 0.04792499542236328}}
3845877266
{'jetson-nano-fp32': {'latency': 0.578345775604248}, 'gtx-1080ti-fp32': {'latency': 0.04792499542236328}}
43100448
0.24906444549560547
0.22275354
```

# Acknowledge
The code is based on https://github.com/SamsungLabs/nb-asr