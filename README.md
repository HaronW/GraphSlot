# GraphSlot

## Introduction

​	Code release for paper : **GraphSlot: An Unsupervised Object Centric Learning with Graph
Representation for Video Object Decomposition**

​	This code contains:

- Training GraphSlot model on MOVi-C dataset
- Evaluate GraphSlot model on MOVi-C dataset
- Checkpoints of GraphSlot on MOVi-C dataset

![figure2](https://github.com/HaronW/GraphSlot/blob/main/figure2.png)



## Installation

​	To setup conda environment, run:

```shell
cd ./GraphSlot
conda env create -f environment.yml
```



## Experiments

#### Data preparation

​	Details about MOVi dataset can be found at [MOVi](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md). The MOVi-C datasets are stored in a [Google Cloud Storage (GCS) bucket](https://console.cloud.google.com/storage/browser/kubric-public/tfds/movi_c) and can be downloaded to local disk prior to training by running:

```shell
cd ./GraphSlot/graphslot/datasets
gsutil -m cp -r "gs://kubric-public/tfds/movi_c/128x128" .
```



#### Train

​	To train GraphSlot, run:

```shell
cd ./GraphSlot
python -m graphslot.main --seed 42 --gpu 0,1,2,3 --mode=graphslot
```



#### Evaluate checkpoints

​	To evaluate GraphSlot, run:

```shell
cd ./GraphSlot
python -m graphslot.main --seed 42 --gpu 0,1,2,3 --mode=graphslot --eval --resume_from ./model/graphslot_100000.pt
```



#### Checkpoints

​	Checkpoint is available at [Google Drive](https://drive.google.com/file/d/10VLmQQ9WUPipAfa_emwBXBEfi4rzAWyo/view?usp=drive_link).



## Acknowledgement

​	We thank the authors of [Slot-Attention](https://github.com/google-research/google-research/tree/master/slot_attention), [SAVi](https://github.com/google-research/slot-attention-video/), and [SAVi-PyTorch](https://github.com/junkeun-yi/SAVi-pytorch) for opening source their wonderful works.



## License

​	GraphSlot is released under the MIT License. See the LICENSE file for more details.