# TC-GNN

### Description

The TC-GNN repository contains the modules of a deep learning pipeline that can pre-process and train a Graph Neural Network on structured data(ERA5 for the atmospheric features, IBTrACS for the ground truth). This is possible by reading the input images like graphs, with pixels treated like nodes and their proximity like edges.
The network can then be used to detect tropical cyclones using a GNN approach instead of the classical CNN.
This repository is also an archive of the experiments that were carried out when working on the data engineering part of this task.

#### The seven ERA5 atmospheric features

<p align="left">
  <img width="900" src=https://github.com/HPCI-Lab/TC-GNN/assets/38779834/daf88c79-ec8c-421f-8d26-6d303e85d27f>
</p>

#### Ground truth vs. predictions(five random density maps at the top and their corresponding predictions at the bottom)

<p align="left">
  <img width="900" src=https://github.com/HPCI-Lab/TC-GNN/assets/38779834/e07bc05c-8793-47d9-8302-f7c72cdcc883>
</p>

### How to Setup

For this pipeline, you need Python already available on your system.

```
$ python3 -m venv ENV_NAME
$ source ./ENV_NAME/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
