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

## Download
- Raw datasets: download [NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) and [highD](https://www.highd-dataset.com/), then process them into the required format (.mat) using the preprocessing [code](https://github.com/Haoran-SONG/PiP-Planning-informed-Prediction/tree/master/preprocess).
- Processed datasets: download from this [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hsongad_connect_ust_hk/Evo9MNDPLhZAn-ygM1-GOVQB-ULdHzx4WurTZ1j-Bk_JNQ?e=YdG7Xk) and save them in datasets/.
- Trained models: download from this [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hsongad_connect_ust_hk/EtttBXgentVNhb3QuYSaK2kBTh0vbL0sno1S3p9bnuKcFA?e=9wb7rQ) and save them in trained_models/.

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


