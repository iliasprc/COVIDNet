import torch.nn as nn

from torchvision import models


class CNN(nn.Module):
    def __init__(self, classes, model='resnet18'):
        super(CNN, self).__init__()
        if (model == 'resnet18'):
            self.cnn = models.resnet18(pretrained=True)
            self.cnn.fc = nn.Linear(512, classes)
        elif (model == 'resnext50_32x4d'):

            self.cnn = models.resnext50_32x4d(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)
        elif (model == 'mobilenet_v2'):

            self.cnn = models.mobilenet_v2(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)

    def forward(self, x):

        return self.cnn(x)
