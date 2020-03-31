# COVIDNet
PyTorch implementation of COVID-Net https://github.com/lindawangg/COVID-Net

Also Google Colab Notebook for plug-n-play training and evaluation

## TO DO

Integration with MedicalZooPytorch https://github.com/black0017/MedicalZooPytorch

Exact implementation of COVID-Net Small and COVID-Net Large as in original paper

Confusion Matrix as in original paper https://arxiv.org/pdf/2003.09871.pdf

## Training and evaluation
The network takes as input an image of shape (N, 224, 224, 3) and outputs the softmax probabilities as (N, 3), where N is the number of batches.
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




## Pretrained Models

My implementation ( Numbers from torchsummary )

| Accuracy (%) | # Params (M) | MACs (G) |        Model        |
|:------------:|:------------:|:--------:|:-------------------:|
|   89.10      |     118.19   |   3.54   |    [COVID-Net]      |

# Links
Check out this repository for more medical applications with deep-learning in PyTorch
https://github.com/black0017/MedicalZooPytorch
