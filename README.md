# Group Sampling
Group Sampling for Unsupervised Person Re-identification

## Requirements

### Installation

```shell
git clone https://github.com/wavinflaghxm/GroupSampling.git
cd GroupSampling
python setup.py develop
```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [MSMT17](https://arxiv.org/abs/1711.08565).
Then unzip them under the directory like:
```
GroupSampling/examples/data
├── market1501
│   └── Market-1501-v15.09.15
├── dukemtmc
│   └── DukeMTMC-reID
└── msmt17
    └── MSMT17_V2
```

## Training

We utilize 1 GTX-1080TI GPUs for training.

+ use `--group-n 256` for Market-1501, `--group-n 128` for DukeMTMC-reID, and `--group-n 2048` for MSMT17;
+ use `--iters 400` (default) for Market-1501 and DukeMTMC-reID, `--iters 800` for MSMT17;

*Market-1501:*
```
python examples/train.py -d market1501 --logs-dir logs/market_resnet50 --group-n 256
```

*DukeMTMC-reID:*
```
python examples/train.py -d dukemtmc --logs-dir logs/duke_resnet50 --group-n 128
```

*MSMT17:*
```
python examples/train.py -d msmt17 --logs-dir logs/msmt_resnet50 --group-n 2048 --iters 800
```

## Evaluation
To evaluate the model, run:
```
python examples/test.py -d $DATASET --resume $PATH

### Market-1501 ###
python examples/test.py -d market1501 --resume logs/market_resnet50/model_best.pth.tar
```


## Results
![results](figs/results.png)

## Acknowledgement

Codes are built upon [SpCL](https://github.com/yxgeee/SpCL).
