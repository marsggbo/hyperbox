
# Run

```bash
python run.py --config-path=weights/OFA  +test_ckpt=weights/OFA/OFA_MBV3_k357_d234_e46.pth
```

# Result

```bash
(nas) [hyperbox@server hyperbox]$ python run.py --config-path=weights/OFA +test_ckpt=weights/OFA/OFA_MBV3_k357_d234_e46.pth
[2021-08-02 08:24:36,276] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/utils/utils.py:91] - Disabling python warnings! <config.ignore_warnings=True>
[2021-08-02 08:24:36,324][pytorch_lightning.utilities.seed][INFO] - Global seed set to 888
[2021-08-02 08:24:36,327] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/train.py:35] - Instantiating datamodule <hyperbox.datamodules.imagenet_datamodule.ImagenetDataModule>
[2021-08-02 08:24:36,328] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/train.py:39] - Instantiating model <hyperbox.models.ofa_model.OFAModel>
[2021-08-02 08:24:36,496] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/models/base_model.py:73] - Building hyperbox.networks.ofa.OFAMobileNetV3 ...
[2021-08-02 08:24:36,498] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/models/base_model.py:80] - Building hyperbox.mutator.random_mutator.RandomMutator ...
[2021-08-02 08:24:36,500] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/models/base_model.py:91] - Building hyperbox.losses.ce_labelsmooth_loss.CrossEntropyLabelSmooth ...
[2021-08-02 08:24:36,501] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/models/base_model.py:103] - Building hyperbox.utils.metrics.Accuracy ...
[2021-08-02 08:24:36,501] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/train.py:47] - Instantiating callback <pytorch_lightning.callbacks.ModelCheckpoint>
[2021-08-02 08:24:36,503] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/train.py:47] - Instantiating callback <pytorch_lightning.callbacks.EarlyStopping>
[2021-08-02 08:24:36,504] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/train.py:55] - Instantiating logger <pytorch_lightning.loggers.wandb.WandbLogger>
[2021-08-02 08:24:36,505] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/train.py:59] - Instantiating trainer <pytorch_lightning.Trainer>
[2021-08-02 08:24:36,645][pytorch_lightning.utilities.distributed][INFO] - GPU available: True, used: True
[2021-08-02 08:24:36,645][pytorch_lightning.utilities.distributed][INFO] - TPU available: False, using: 0 TPU cores
[2021-08-02 08:24:36,646][pytorch_lightning.utilities.distributed][INFO] - IPU available: False, using: 0 IPUs
[2021-08-02 08:24:36,647] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/train.py:65] - Logging hyperparameters!
wandb: W&B syncing is set to `offline` in this directory.  Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
[2021-08-02 08:24:48,059][pytorch_lightning.accelerators.gpu][INFO] - LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
[2021-08-02 08:26:11,602] [INFO] [/home/xihe/xinhe/hyperbox/hyperbox/models/ofa_model.py:241] - Test epoch0 final result: loss=1.889573727422358, acc=77.36373081841433
```