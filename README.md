# COVIDNet

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IliasPap/COVIDNet/blob/master/COVIDNet.ipynb#scrollTo=G9t--VlR7_F6)
My PyTorch implementation of COVID-Net, for the original work please see: https://github.com/lindawangg/COVID-Net

The purpose of this github is to reproduce results and not to claim state-of-the-art performance !!

Also Google Colab Notebook for plug-n-play training and evaluation here [![Open In Colab](https://colab.research.google.com/github/IliasPap/COVIDNet/blob/master/COVIDNet.ipynb#scrollTo=G9t--VlR7_F6)


## Table of Contents

* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [Datasets](#datasets)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)


<!-- GETTING STARTED -->
## Getting Started



## TODOs

- [ ] Final Requirements
- [ ] Pretrained models
- [ ] Test all pretrained models
- [ ] Instructions for training 
- [ ] Adding command line option for inference

## Requirements

### Installation & Data Preparation

Please refer to 
```python
 pip install -r requirements.txt
```




* Python >= 3.6 (3.6 recommended)
* PyTorch >= 1.4 (1.6.0 recommended)
* torchvision >=0.6.0  
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 
<!-- USAGE EXAMPLES -->
## Usage

### Training

The network takes as input an image of shape (N, 224, 224, 3) and outputs the softmax probabilities as (N, C), where N is the number of batches and C number of output classes.

1. To train the Network from scratch simply do `python main.py` 
 Arguments for training 
```yaml
trainer:
  cwd: /home/ # working directory
  logger: CovidCLF # logger name
  epochs: 30 # number of training epochs
  seed: 123 # randomness seed
  cuda: True # use nvidia gpu
  gpu: 0,1 # id of gpu
  save: True # save checkpoint
  load: False # load pretrained checkpoint
  gradient_accumulation: 1 # gradient accumulation steps
  pretrained_cpkt: cpkt.pt
  log_interval: 1000 # print statistics every log_interval
  model:
    name: mobilenet_v2 # model name  [mobilenet_v2,COVIDNet_small]
    optimizer: # optimizer configuration
      type: SGD # optimizer type
      lr: 1e-2 # learning rate
      weight_decay: 0.000001 # weight decay
    scheduler: # learning rate scheduler
      type: ReduceLRonPlateau # type of scheduler
      scheduler_factor: 0.5 # learning rate change ratio
      scheduler_patience: 0 # patience for some epochs
      scheduler_min_lr: 1e-3 # minimum learning rate value
      scheduler_verbose: 5e-6 # print if learning rate is changed
  dataloader:
    train:
      batch_size: 4 # batch size
      shuffle: True # shuffle samples after every epoch
      num_workers: 2 # number of thread for dataloader1
    val:
      batch_size: 2
      shuffle: False
      num_workers: 2
    test:
      batch_size: 1
      shuffle: False
      num_workers: 2
  dataset:
    input_data: ./data/data
    name: COVIDx # dataset name COVIDx or COVID_CT
    modality: RGB # type of modality
    dim: [224,224] # image dimension
    train:
      augmentation: True # do augmentation to video
    val:
      augmentation: False
    test:
      augmentation: False
```

<!-- RESULTS -->
## Results 


with my   implementation  of COVID-Net and comparison with CNNs pretrained on ImageNet dataset


### Results in COVIDx  dataset 


|    Model        | Accuracy (%) | # Params (M) | MACs (G) |      
|:------------:|:------------:|:--------:|:-------------------:|
 | [COVID-Net-Small] |    |   89.10      |     115.42   |   2.26   |  
 |   [COVID-Net-Large](https://drive.google.com/open?id=1-3SKFua_wFl2_aAQMIrj2FhowTX8B551) |   91.22      |     118.19   |   3.54   | 
 |   [Mobilenet V2   ](https://drive.google.com/open?id=19J-1bW6wPl7Kmm0pNagehlM1zk9m37VV) |   94.0       |     -   |   -      |
 |   [ResNeXt50-32x4d](https://drive.google.com/open?id=1-BLolPNYMVWSY0Xnm8Y8wjQCapXiPnLx) |   95.0       |     -   |   -      |
 | [ResNet-18](https://drive.google.com/open?id=1wxo4gkNGyrhR-1PG8Vr1hj65MfSAHOgJ)   |   94.0       |     -   |   -       |

### Results in COVID-CT  dataset 


|  Model       | Accuracy (%) | # Params (M) | MACs (G) | 
|:------------:|:------------:|:--------:|:-------------------:|
|    [COVID-Net-Small]  |     -   |  -   |  |
|     [COVID-Net-Large]     |     -   |   -  |  |


Confusion Matrix on both datasets coming soon !!




<!-- Datasets -->
## Datasets
### 1) COVID-CT-Dataset

The COVID-CT-Dataset has 288 CT images containing clinical findings of COVID-19. We are continuously adding more COVID CTs.

The images are collected from medRxiv and bioRxiv papers about COVID-19. CTs containing COVID-19 abnormalities are selected by reading the figure captions in the papers. All copyrights of the data belong to medRxiv and bioRxiv.

Please refer to the preprint for details: COVID-CT-Dataset: A CT Scan Dataset about COVID-19

### 2) COVIDx  dataset 


The current COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

We especially thank the Radiological Society of North America and others involved in the RSNA Pneumonia Detection Challenge, and Dr. Joseph Paul Cohen and the team at MILA involved in the COVID-19 image data collection project, for making data available to the global community.

### Steps to generate the COVIDx dataset

Download the datasets listed above
 * `git clone https://github.com/ieee8023/covid-chestxray-dataset.git`
 * go to this [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) to download the RSNA pneumonia dataset
2. Create a `data` directory and within the data directory, create a `train` and `test` directory
3. Use [COVIDNet.ipynb](COVIDNet.ipynb) to combine the two dataset to create COVIDx. Make sure to remember to change the file paths.
4. We provide the train and test txt files with patientId, image path and label (normal, pneumonia or COVID-19). The description for each file is explained below:
 * [train\_COVIDx.txt](train_COVIDx.txt): This file contains the samples used for training.
 * [test\_COVIDx.txt](test_COVIDx.txt): This file contains the samples used for testing.







[contributors-shield]: https://img.shields.io/github/contributors/iliasprc/COVIDNet.svg?style=flat-square
[contributors-url]: https://github.com/iliasprc/COVIDNet/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/iliasprc/COVIDNet.svg?style=flat-square
[forks-url]: https://github.com/iliasprc/COVIDNet/network/members

[stars-shield]: https://img.shields.io/github/stars/iliasprc/COVIDNet.svg?style=flat-square
[stars-url]: https://github.com/iliasprc/COVIDNet/stargazers

[issues-shield]: https://img.shields.io/github/issues/iliasprc/COVIDNet.svg?style=flat-square
[issues-url]: https://github.com/iliasprc/COVIDNet/issues





# Links
Check out this repository for more medical applications with deep-learning in PyTorch
https://github.com/black0017/MedicalZooPytorch from https://github.com/black0017
