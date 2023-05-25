from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flattens the input tensor for use in the neural network.

    This class defines the operation to flatten an input tensor.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Flattens the input tensor.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The flattened tensor.
        """
        return input.view(input.size(0), -1)


class PEPX(nn.Module):
    """Defines the PEPX Convolutional Neural Network architecture.

    This class creates a PEPX model, which contains several convolutional layers
    for feature projection, expansion, depth-wise representation, second-stage
    projection, and final extension.

    Args:
        n_input (int): The number of input features.
        n_out (int): The number of output features.

    Attributes:
        network (nn.Sequential): The neural network layers.
    """

    def __init__(self, n_input: int, n_out: int):

        self.network = nn.Sequential(nn.Conv2d(in_channels=n_input, out_channels=n_input // 2, kernel_size=1),
                                     nn.Conv2d(in_channels=n_input // 2, out_channels=int(3 * n_input / 4),
                                               kernel_size=1),
                                     nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=int(3 * n_input / 4),
                                               kernel_size=3, groups=int(3 * n_input / 4), padding=1),
                                     nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=n_input // 2,
                                               kernel_size=1),
                                     nn.Conv2d(in_channels=n_input // 2, out_channels=n_out, kernel_size=1),
                                     nn.BatchNorm2d(n_out))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the PEPX model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor, result of the forward pass of the model.
        """
        return self.network(x)


class CovidNet(nn.Module):
    """Defines the CovidNet model architecture for classification tasks.

    The architecture consists of a sequence of PEPX blocks and linear layers. It can be built
    in a small or large configuration depending on the model parameter.

    Args:
        model (str): Model type, either 'small' or 'large'. Defaults to 'small'.
        n_classes (int): Number of classes for the final classification. Defaults to 3.

    Attributes:
        All the PEPX blocks and other layers in the network.
    """

    def __init__(self, model: str = 'small', n_classes: int = 3):
        super(CovidNet, self).__init__()

        filters = {
            'pepx1_1': [56, 56],
            'pepx1_2': [56, 56],
            'pepx1_3': [56, 56],
            'pepx2_1': [56, 112],
            'pepx2_2': [112, 112],
            'pepx2_3': [112, 112],
            'pepx2_4': [112, 112],
            'pepx3_1': [112, 216],
            'pepx3_2': [216, 216],
            'pepx3_3': [216, 216],
            'pepx3_4': [216, 216],
            'pepx3_5': [216, 216],
            'pepx3_6': [216, 216],
            'pepx4_1': [216, 424],
            'pepx4_2': [424, 424],
            'pepx4_3': [424, 424],
        }

        self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=56, kernel_size=7, stride=2, padding=3))
        for key in filters:

            if ('pool' in key):
                self.add_module(key, nn.MaxPool2d(filters[key][0], filters[key][1]))
            else:
                self.add_module(key, PEPX(filters[key][0], filters[key][1]))

        if (model == 'large'):

            self.add_module('conv1_1x1', nn.Conv2d(in_channels=56, out_channels=112, kernel_size=1))
            self.add_module('conv2_1x1', nn.Conv2d(in_channels=112, out_channels=216, kernel_size=1))
            self.add_module('conv3_1x1', nn.Conv2d(in_channels=216, out_channels=424, kernel_size=1))
            self.add_module('conv4_1x1', nn.Conv2d(in_channels=424, out_channels=424, kernel_size=1))

            self.__forward__ = self.forward_large_net
        else:
            self.__forward__ = self.forward_small_net
        self.add_module('flatten', Flatten())
        self.add_module('fc1', nn.Linear(7 * 7 * 424, 512))

        self.add_module('classifier', nn.Linear(512, n_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.__forward__(x)

    def forward_large_net(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Defines the forward pass for the large model variant.

        Args:
            x (torch.Tensor): The input tensor.
            target (torch.Tensor, optional): The target tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor, result of the forward pass of the large model.
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out_conv1_1x1 = self.conv1_1x1(x)

        pepx11 = self.pepx1_1(x)
        pepx12 = self.pepx1_2(pepx11 + out_conv1_1x1)
        pepx13 = self.pepx1_3(pepx12 + pepx11 + out_conv1_1x1)

        out_conv2_1x1 = F.max_pool2d(self.conv2_1x1(pepx12 + pepx11 + pepx13 + out_conv1_1x1), 2)

        pepx21 = self.pepx2_1(
            F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2) + F.max_pool2d(out_conv1_1x1,
                                                                                                       2))
        pepx22 = self.pepx2_2(pepx21 + out_conv2_1x1)
        pepx23 = self.pepx2_3(pepx22 + pepx21 + out_conv2_1x1)
        pepx24 = self.pepx2_4(pepx23 + pepx21 + pepx22 + out_conv2_1x1)

        out_conv3_1x1 = F.max_pool2d(self.conv3_1x1(pepx22 + pepx21 + pepx23 + pepx24 + out_conv2_1x1), 2)

        pepx31 = self.pepx3_1(
            F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23,
                                                                                                       2) + F.max_pool2d(
                out_conv2_1x1, 2))
        pepx32 = self.pepx3_2(pepx31 + out_conv3_1x1)
        pepx33 = self.pepx3_3(pepx31 + pepx32 + out_conv3_1x1)
        pepx34 = self.pepx3_4(pepx31 + pepx32 + pepx33 + out_conv3_1x1)
        pepx35 = self.pepx3_5(pepx31 + pepx32 + pepx33 + pepx34 + out_conv3_1x1)
        pepx36 = self.pepx3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + out_conv3_1x1)

        out_conv4_1x1 = F.max_pool2d(
            self.conv4_1x1(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + pepx36 + out_conv3_1x1), 2)

        pepx41 = self.pepx4_1(
            F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34,
                                                                                                       2) + F.max_pool2d(
                pepx35, 2) + F.max_pool2d(pepx36, 2) + F.max_pool2d(out_conv3_1x1, 2))
        pepx42 = self.pepx4_2(pepx41 + out_conv4_1x1)
        pepx43 = self.pepx4_3(pepx41 + pepx42 + out_conv4_1x1)
        flattened = self.flatten(pepx41 + pepx42 + pepx43 + out_conv4_1x1)

        fc1out = F.relu(self.fc1(flattened))

        logits = self.classifier(fc1out)
        return logits

    def forward_small_net(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for the small model variant.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor, result of the forward pass of the small model.
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        pepx11 = self.pepx1_1(x)
        pepx12 = self.pepx1_2(pepx11)
        pepx13 = self.pepx1_3(pepx12 + pepx11)

        pepx21 = self.pepx2_1(F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2))
        pepx22 = self.pepx2_2(pepx21)
        pepx23 = self.pepx2_3(pepx22 + pepx21)
        pepx24 = self.pepx2_4(pepx23 + pepx21 + pepx22)

        pepx31 = self.pepx3_1(
            F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23, 2))
        pepx32 = self.pepx3_2(pepx31)
        pepx33 = self.pepx3_3(pepx31 + pepx32)
        pepx34 = self.pepx3_4(pepx31 + pepx32 + pepx33)
        pepx35 = self.pepx3_5(pepx31 + pepx32 + pepx33 + pepx34)
        pepx36 = self.pepx3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35)

        pepx41 = self.pepx4_1(
            F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34,
                                                                                                       2) + F.max_pool2d(
                pepx35, 2) + F.max_pool2d(pepx36, 2))
        pepx42 = self.pepx4_2(pepx41)
        pepx43 = self.pepx4_3(pepx41 + pepx42)
        flattened = self.flatten(pepx41 + pepx42 + pepx43)

        fc1out = F.relu(self.fc1(flattened))
        # fc2out = F.relu(self.fc2(fc1out))
        logits = self.classifier(fc1out)
        return logits
