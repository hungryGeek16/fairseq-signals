# Fairseq-signals

Fairseq-signals is a collection of deep learning models for ECG data processing based on the [`fairseq`](https://github.com/pytorch/fairseq).

We provide implementations of various deep learning methods on ECG data, including official implementations of our works.

### Implemented Paper:
* [Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram](https://arxiv.org/abs/2203.06889)*

# Requirements and Installation
* [PyTorch](https://pytorch.org) version == 1.13.1+cu117
* Python version == 3.9
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq-signals** from source and develop locally:

```bash
git clone https://github.com/Jwoo5/fairseq-signals
cd fairseq-signals
pip install --editable ./
pip install scikit-learn transformers
```

* **To preprocess ECG datasets**: `pip install scipy wfdb`
* **To build cython components**: `python setup.py build_ext --inplace`
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`

# Getting Started
## For uni-modal tasks (ECG Classification, ...)
### Prepare ECG dataset
We provide pre-processing codes for various ECG datasets.

* [PhysioNet2021](https://drive.google.com/file/d/1STzmvuM8Jeo71n-drcthmbXe0Du5eSQk/view?usp=sharing)
* The dataset must be in the given form:
```bash
- physionet_org
      |===> training
               |----> chapman_shaoxing		
               |----> cpsc_2018		
               |----> cpsc_2018_extra		
               |----> georgia		
               |----> ningbo		
               |----> ptb		
               |----> ptb-xl		
               |----> st_petersburg_incart
      |===> dx_mapping_scored.csv
      |===> dx_mapping_unscored.csv
      |===> evaluate_model.m
      |===> evaluate_model.py
      |===> helper_code.py
      |===> weights.csv
      |===> weights_abbreviations.csv
``` 

### Pre-process
Given a directory that contains WFDB directories to be pre-processed for **PhysioNet2021**:

```shell script
$ python fairseq_signals/data/ecg/preprocess/preprocess_physionet2021.py \
    /path/to/physionet2021/ \
    --dest /path/to/output \
    --workers $N
```

### Prepare data manifest
Given a directory that contains pre-processed data:
```shell script
$ python fairseq_signals/data/ecg/preprocess/manifest.py \
    /path/to/data/ \
    --dest /path/to/manifest \
    --valid-percent $valid
```
# Prepare training data manifest
Before training, you should prepare training data manifest required for training CLOCS model.
```shell script
$ python /path/to/fairseq_signals/data/ecg/preprocess/convert_to_clocs_manifest.py \
    /path/to/pretrain/train.tsv \
    --dest /path/to/manifest
```
The expected results are like:
```
/path/to/manifest
├─ cmsc
│  └─ train.tsv
├─ cmlc
│  └─ train.tsv
└─ cmsmlc
   └─ train.tsv
```

# Pre-training a new model
This configuration was used for the `W2V+CMSC+RLM` model pre-trained on the `PhysioNet2021` dataset in the original paper.

```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest/cmsc \ #Absolute path in windows
    --config-dir examples/w2v_cmsc/config/pretraining \
    --config-name w2v_cmsc_rlm
```
