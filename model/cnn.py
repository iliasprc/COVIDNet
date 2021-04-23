import torch.nn as nn

from torchvision import models


class CNN(nn.Module):
    def __init__(self, classes, model='resnet50'):
        super(CNN, self).__init__()
        if (model == 'resnet50'):
            self.cnn = models.resnet50(pretrained=True)
            self.cnn.fc = nn.Linear(2048, classes)
        elif (model == 'resnext50_32x4d'):

            self.cnn = models.resnext50_32x4d(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)
        elif (model == 'mobilenet_v2'):

            self.cnn = models.mobilenet_v2(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)
        elif (model == 'densenet121'):
            self.cnn = models.densenet121(pretrained=True)
            self.cnn.classifier = nn.Linear(1024, classes)

    def forward(self, x):

        return self.cnn(x)
