# EPN
The implementation of "EPN: An Ego Vehicle Planning-Informed Network for Target Trajectory Prediction"

## Dependencies

```shell
conda create -n EPN python=3.7
source activate EPN

conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install tensorboard=1.14.0
conda install numpy=1.16 scipy=1.4 h5py=2.10 future
```

## Dataset
Please refer to this [website](https://github.com/Haoran-SONG/PiP-Planning-informed-Prediction) for data processing

## Running

Training by `sh scripts/train.sh` or running
```shell
python train.py --name ngsim_demo --batch_size 64 --pretrain_epochs 5 --train_epochs 10 --train_set ./datasets/NGSIM/train.mat --val_set ./datasets/NGSIM/val.mat 
```


Test by `sh scripts/test.sh` or running
```shell
python evaluate.py --name ngsim_model --batch_size 64 \
    --test_set ./datasets/NGSIM/test.mat
```


## Acknowledgments
We would like to express our gratitude to the authors of the following repository for their excellent work and contributions, which greatly inspired this project:
- [PiP](https://github.com/Haoran-SONG/PiP-Planning-informed-Prediction)

