from abc import abstractmethod

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """

        Args:
            *inputs (torch.Tensor): input data to model
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, train_batch, batch_idx=None):
        """
        training function of vodel
        Args:
            train_batch (tuple): (data, target)
            batch_idx (int):
        """
        pass

    @abstractmethod
    def validation_step(self, train_batch, batch_idx=None):
        """
        validation function of vodel
        Args:
            train_batch (tuple): (data, target)
            batch_idx (int):
        """
        pass

    @abstractmethod
    def loss(self, *inputs):
        """
        Loss calculation
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
