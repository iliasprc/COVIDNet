# COVIDNet

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

My PyTorch implementation of COVID-Net, for the original work please see: https://github.com/lindawangg/COVID-Net

The purpose of this github is to reproduce results and not to claim state-of-the-art performance !!

Also Google Colab Notebook for plug-n-play training and evaluation here -->[IliasPap/covidnet.ipynb](https://gist.github.com/IliasPap/598e93ec50fe84f7953eef359d715916)

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

### Installation
To install the required python packages use the following command 
```
pip install -r requirements.txt
```
<!-- USAGE EXAMPLES -->
## Usage

### Training

The network takes as input an image of shape (N, 224, 224, 3) and outputs the softmax probabilities as (N, C), where N is the number of batches and C number of output classes.

1. To train the Network from scratch simply do `python main.py` 
 Arguments for training 
 ```
   -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size foe training
  --log_interval LOG_INTERVAL
                        steps to print metrics and loss
  --dataset_name DATASET_NAME
                        dataset name
  --nEpochs NEPOCHS     total number of epochs
  --device DEVICE       gpu device
  --seed SEED           select seed number for reproducibility
  --classes CLASSES     dataset classes
  --lr LR               learning rate (default: 1e-3)
  --weight_decay WEIGHT_DECAY
                        weight decay (default: 1e-6)
  --cuda                use gpu for speed-up
  --tensorboard         use tensorboard for loggging and visualization
  --resume PATH         path to latest checkpoint (default: none)
  --model {COVIDNet_small,resnet18,mobilenet_v2,densenet169,COVIDNet_large}
  --opt {sgd,adam,rmsprop}
  --root_path ROOT_PATH
                        path to dataset
  --save SAVE           path to checkpoint save directory


```
<!-- RESULTS -->
## Results 


with my   implementation  of COVID-Net and comparison with CNNs pretrained on ImageNet dataset


### Results in COVIDx  dataset 


| Accuracy (%) | # Params (M) | MACs (G) |        Model        |
|:------------:|:------------:|:--------:|:-------------------:|
|   89.10      |     115.42   |   2.26   |   [COVID-Net-Small] |
|   91.22      |     118.19   |   3.54   |   [COVID-Net-Large](https://drive.google.com/open?id=1-3SKFua_wFl2_aAQMIrj2FhowTX8B551) |
|   94.0       |     -   |   -      |   [Mobilenet V2   ](https://drive.google.com/open?id=19J-1bW6wPl7Kmm0pNagehlM1zk9m37VV) |
|   95.0       |     -   |   -      |   [ResNeXt50-32x4d](https://drive.google.com/open?id=1-BLolPNYMVWSY0Xnm8Y8wjQCapXiPnLx) |
|   94.0       |     -   |   -      | [ResNet-18](https://drive.google.com/open?id=1wxo4gkNGyrhR-1PG8Vr1hj65MfSAHOgJ) |

### Results in COVID-CT  dataset 
Soon ...

| Accuracy (%) | # Params (M) | MACs (G) |        Model        |
|:------------:|:------------:|:--------:|:-------------------:|
|   -   |     -   |  -   |   [COVID-Net-Small] |
|   -      |     -   |   -  |   [COVID-Net-Large] |
|   76      |     -   |   -      |   [Mobilenet V2   ](https://drive.google.com/open?id=1alVSSN-PkibfFQcH0RA1xIPMSfbVxI89) |
|   76    |     -   |   -      |   [ResNeXt50-32x4d] |
|  73     |     -   |   -      | [ResNet-18] |
|  81    |     -   |   -      | [Densenet-169] |

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


Chest radiography images distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    8514   |    66    | 16546 |
|  test |   100  |     100   |    10    |   210 |







[contributors-shield]: https://img.shields.io/github/contributors/IliasPap/COVIDNet.svg?style=flat-square
[contributors-url]: https://github.com/IliasPap/COVIDNet/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/IliasPap/COVIDNet.svg?style=flat-square
[forks-url]: https://github.com/IliasPap/COVIDNet/network/members

[stars-shield]: https://img.shields.io/github/stars/IliasPap/COVIDNet.svg?style=flat-square
[stars-url]: https://github.com/IliasPap/COVIDNet/stargazers

[issues-shield]: https://img.shields.io/github/issues/IliasPap/COVIDNet.svg?style=flat-square
[issues-url]: https://github.com/IliasPap/COVIDNet/issues





# Links
Check out this repository for more medical applications with deep-learning in PyTorch
https://github.com/black0017/MedicalZooPytorch from https://github.com/black0017
