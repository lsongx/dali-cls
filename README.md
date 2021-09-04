
# How to use

1. Update your cluster setting in `submit.py`.
2. Submit your job with `python3 submit.py -s -i ini/mbnet_v1_ce.ini`.

## For `cuda9.2`

```[shell]
hdfs dfs -get /user/liangchen.song/data/conda_lib9_torch11_dali.tar
tar xf conda_lib9_torch11_dali.tar
./conda/bin/python -m torch.distributed.launch --nproc_per_node=2 train.py [args]
```

Please see `py_job.sh` for example.

# 实测结果

价格根据 [wiki上](http://wiki.hobot.cc/pages/viewpage.action?pageId=63999389) 算出 `base 2.31, Titan V *1.82, 2080Ti *1.58`.

|网络|设置|batch|Epochs|集群|时间(h)|价格|结果|备注|
|:-:|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|mbv1|cosine/epoch|8*128|120|IDC|15.5|286.44|72.65||
|mbv1|cosine/step|8*128|120|IDC|15.5|286.44|72.43||
|mbv1|cosine/epoch, FP16|2*150|120|金山云|30|252.25|72.52||
|mbv1|cosine/step, FP16, contrast -0.2|2*128|120|2080Ti|35|255.49|72.53||
|mbv1|cosine/step|2*128|120|IDC|58|267.96|72.36||
|mbv1|color twist +/-0.4|8*128|120|IDC|15|277.2|__72.72__|ID: 85764|
|mbv1|all checked|8*128|240|IDC|31|572.88|72.99||
|mbv1|color twist +/-0.4|8*128|240|2080Ti|18.3|535.21|73.08|最好版本了|
|res50|wd 4e-5, contrast -0.2|8*64|120|IDC|33.5|619.08|76.09|精度太低|
|res50|wd 4e-5, contrast -0.2|8*64|120|阿里云|34.5|637.56|76.04|精度太低|
|res50|wd 1e-4, color twist +/-0.4|4*64|120|IDC|70|646.80|76.84|超过官方76.15|

# TODO List

- [] jpeg decoder有问题, 可能是这个引起的掉点. [wiki说明](http://wiki.hobot.cc/pages/viewpage.action?pageId=68301317)

# Bugs

Some unsolved bugs.

## 1. 阿里云small, 2卡mbv1, 每卡150 batch, 在第二个epoch时出错

```[]
Traceback (most recent call last):
  File "train.py", line 122, in <module>
    main()
  File "train.py", line 118, in main
    logger=logger)
  File "/running_package/torch_submit/mmcls/apis/train.py", line 57, in train_model
    _dist_train(model, cfg, validate=validate, logger=logger)
  File "/running_package/torch_submit/mmcls/apis/train.py", line 177, in _dist_train
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
  File "/running_package/torch_submit/conda/lib/python3.6/site-packages/mmcv/runner/runner.py", line 358, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/running_package/torch_submit/conda/lib/python3.6/site-packages/mmcv/runner/runner.py", line 260, in train
    for i, data_batch in enumerate(data_loader):
  File "/running_package/torch_submit/conda/lib/python3.6/site-packages/nvidia/dali/plugin/pytorch.py", line 127, in __next__
    outputs.append(p._share_outputs())
  File "/running_package/torch_submit/conda/lib/python3.6/site-packages/nvidia/dali/pipeline.py", line 294, in _share_outputs
    return self._pipe.ShareOutputs()
RuntimeError: Critical error in pipeline: CUDA allocation failed
Current pipeline object is no longer valid.
```

## 2. IDC small, 2卡mbv1, 每卡150 batch, 在第一个epoch时出错

```[]
Traceback (most recent call last):
  File "train.py", line 122, in <module>
    main()
  File "train.py", line 118, in main
    logger=logger)
  File "/running_package/torch_submit/mmcls/apis/train.py", line 57, in train_model
    _dist_train(model, cfg, validate=validate, logger=logger)
  File "/running_package/torch_submit/mmcls/apis/train.py", line 177, in _dist_train
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
  File "/running_package/torch_submit/conda/lib/python3.6/site-packages/mmcv/runner/runner.py", line 358, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/running_package/torch_submit/conda/lib/python3.6/site-packages/mmcv/runner/runner.py", line 260, in train
    for i, data_batch in enumerate(data_loader):
  File "/running_package/torch_submit/conda/lib/python3.6/site-packages/nvidia/dali/plugin/pytorch.py", line 127, in __next__
    outputs.append(p._share_outputs())
  File "/running_package/torch_submit/conda/lib/python3.6/site-packages/nvidia/dali/pipeline.py", line 294, in _share_outputs
    return self._pipe.ShareOutputs()
RuntimeError: Critical error in pipeline: std::bad_alloc
Current pipeline object is no longer valid.
```

__batch size 设为128时能跑, 怀疑是显存不够时报的错. 但是不用dali时是可以跑150 batch size的. 似乎dali需要更多显存.__



```
pip install --upgrade --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110==0.28.0
```