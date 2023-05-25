import torch
import torch.nn as nn

from torchvision import models


class CNN(nn.Module):
    """Convolutional Neural Network (CNN) model class.

    This class allows creation of a Convolutional Neural Network (CNN) model.
    The model can be based on ResNet50, ResNeXt50, MobileNetV2, or DenseNet121 architectures,
    and it is fine-tuned for a custom number of output classes.

    Attributes:
        cnn (nn.Module): The underlying model, chosen among available options and pretrained.

    Args:
        classes (int): The number of output classes.
        model (str, optional): The name of the base model to use.
            Options are 'resnet50', 'resnext50_32x4d', 'mobilenet_v2', and 'densenet121'. Default is 'resnet50'.
    """
    def __init__(self, classes: int, model: str = 'resnet50'):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the CNN model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor, result of the forward pass of the model.
        """
        return self.cnn(x)