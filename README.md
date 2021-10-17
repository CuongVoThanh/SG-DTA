<div align="center">
  
# SG-DTA: Stacked Graph Drug-Target Association
<img src="reference/architecture_model.png" alt="Model Overview" width="600" height="400"/>
</div>

**Note (2021-17-10)**: We will public the dataset for this repository after our research is accepted.

### Table of contents
1. [Prerequisites](#1-prerequisites)
2. [Install](#2-install)
3. [Getting started](#3-getting-started)
4. [Results](#4-results)
5. [Repo structure](#5-repo-structure)
<!-- 6. [Citation](#6-Citation) -->

### 1. Prerequisites
- Python == 3.8 is required will all installed including Pytorch>=1.7 with CUDA 11.0 (When you run with other verions, the results might be slightly different).
- Pytorch Geometric for Pytorch == 1.7.0: Please check be careful to match the device used when installing.
- rdkit == 2020.09.1: Generate the compound network.
- Other dependencies are described in `requirements.txt`

### 2. Install
- Creating conda environment for the experiment:
```bash
conda create -n sgdta python=3.8 -y
conda activate sgdta
```
- Installing PyTorch, Torchvision and Pytorch Geometric depending on the device you use to run the experiment:  
The following setting, we config environment for CPU and GPU device with Pytorch == 1.7.0, CUDA 11.0.

**For CPU version**
```bash
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f 
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-geometric==1.6.3 --use-feature=2020-resolver
```

**For GPU version**
```bash
pip install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric==1.6.3 --use-feature=2020-resolver
```

- Installing **rdkit** and other dependencies:
```bash
conda install -y -c conda-forge rdkit==2020.09.1
pip install -r req.txt
```

### 3. Getting started
#### Setting Datasets
(We will release soon)

#### Training and Evaluation
- Training command line:
  ```bash
  python main.py train --train_fold ${fold} --dataset ${dataset} --drug_embedding ${drug_model} --protein_embedding ${protein_model} --network_embedding ${node_embedding_model} --model ${model} --data_type ${data_type} --exp_name ${experiment_name}
  ```
  ...

  - Training a model from scratch:
    
    For example, 
    
    To train model GraphDTA with all KIBA data training and use gat-gcn for drug embedding:
    ```bash
    python main.py train --train_fold 6 --dataset kiba --drug_embedding gat_gcn --protein_embedding embedding --model graphdta --model_data dataDTA --exp_name "Graph_kiba_gat_gcn"
    ```
    To train model SG-DTA with all KIBA data training, and gat-gcn for drug embedding, CNN for target protein, GCN for drug-target network:
    ```bash
    python main.py train --train_fold 6 --dataset kiba --drug_embedding gat_gcn --protein_embedding embedding --model graphdta --network_embedding gcn --model_data dataDTA --exp_name "SGDTA_kiba_gat_gcn_GCN"
    ```

  - Training a model using pretrained_weight: Please config the "exp_name" equivalently to the folder name downloaded from google drive.
    
    For example,
    
    To train model GraphDTA with all KIBA data and gat_gcn for drug embedding using pretrained "Graph_kiba_gat_gcn"
    ```bash
    python main.py train --resume_fold 6  --dataset kiba --drug_embedding gat_gcn --protein_embedding embedding --model graphdta --data_type dataDTA --exp_name "Graph_kiba_gat_gcn"
    ```
    To train model SG-DTA with all KIBA data and gat-gcn for drug embedding, CNN for target protein, GCN for drug-target network
    ```bash
    python main.py train --resume_fold 6 --dataset kiba --drug_embedding gat_gcn --protein_embedding embedding --model graphdta --network_embedding gcn --model_data dataDTA --exp_name "SGDTA_kiba_gat_gcn_GCN"
    ```

- Evaluating command line:
  ```bash
  python main.py test --test_on_fold ${fold} --dataset {dataset} --drug_embedding ${drug_model} --protein_embedding ${protein_model} --network_embedding ${node_embedding_model} --model ${model} --data_type dataDTA --exp_name ${experiment_name}
  ```
  ...
  
  In order to evaluate model, please consistent the "exp_name" equivalently to the folder name of trained model.
  
  For example,
  
  To evaluate DeepDTA model with DAVIS dataset, and CNN for drug_embedding,  CCN for protein_embedding:
  ```bash
  python main.py test --test_on_fold 6 --dataset davis --graph_embedding embedding --protein_embedding embeddingdeep --network_embedding gcn --model deepdta --data_type dataDTA --exp_name "Deep_davis_fulldata"
  ```
  
  To evaluate SG-DTA model with DAVIS dataset, and CNN for drug, CNN for protein, GCN for drug_target network:
  ```bash
  python main.py test --test_on_fold 6 --dataset davis --graph_embedding embedding --protein_embedding embeddingdeep --network_embedding gcn --model sgdta --data_type dataDTA --exp_name "Deep_davis_gcn_fulldata"
  ```

