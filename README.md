# COVIDNet
My PyTorch implementation of COVID-Net, for the original work please see: https://github.com/lindawangg/COVID-Net

Also Google Colab Notebook for plug-n-play training and evaluation




## Models 

My implementation ( Numbers from my training )

| Accuracy (%) | # Params (M) | MACs (G) |        Model        |
|:------------:|:------------:|:--------:|:-------------------:|
|   89.10      |     115.42   |   2.26   |   [COVID-Net-Small] |
|   91.22      |     118.19   |   3.54   |   [COVID-Net-Large] |

## Ablation Study
Comparison with CNNs pretrained on ImageNet dataset



| Accuracy (%) | # Params (M) | MACs (G) |        Model        |
|:------------:|:------------:|:--------:|:-------------------:|
|   94.0       |     -   |   -      |   [Mobilenet V2   ](https://drive.google.com/open?id=19J-1bW6wPl7Kmm0pNagehlM1zk9m37VV) |
|   95.0       |     -   |   -      |   [ResNeXt50-32x4d](https://drive.google.com/open?id=1-BLolPNYMVWSY0Xnm8Y8wjQCapXiPnLx) |
|   94.0       |     -   |   -      | [ResNet-18](https://drive.google.com/open?id=1wxo4gkNGyrhR-1PG8Vr1hj65MfSAHOgJ) |


## TO DO

Integration with MedicalZooPytorch: https://github.com/black0017/MedicalZooPytorch 

Confusion Matrix as in original paper: https://arxiv.org/pdf/2003.09871.pdf

## Training and evaluation
The network takes as input an image of shape (N, 224, 224, 3) and outputs the softmax probabilities as (N, 3), where N is the number of batches.

### COVIDx  dataset 


The current COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

We especially thank the Radiological Society of North America and others involved in the RSNA Pneumonia Detection Challenge, and Dr. Joseph Paul Cohen and the team at MILA involved in the COVID-19 image data collection project, for making data available to the global community.

### Steps to generate the dataset

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


### Steps for training
1. To train from scratch simply do `python main.py` 
2. For argument options  `python main.py --help` 

### Steps for inference
Releasing soon

## Results
Confusion Matrix Coming soon !!

## Requirements

Python > 3.5

PyTorch > 1.0

Numpy



# Links
Check out this repository for more medical applications with deep-learning in PyTorch
https://github.com/black0017/MedicalZooPytorch from https://github.com/black0017
