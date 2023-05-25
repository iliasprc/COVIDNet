import torch.nn.functional as F

import torch
def nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the negative log-likelihood loss.

    Args:
        output (torch.Tensor): The predicted output from the model.
        target (torch.Tensor): The true labels.

    Returns:
        torch.Tensor: The computed negative log-likelihood loss.
    """
    return F.nll_loss(output, target)


def crossentropy_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the cross-entropy loss.

    Args:
        output (torch.Tensor): The predicted output from the model.
        target (torch.Tensor): The true labels.

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    return F.cross_entropy(output, target)


def focal_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the focal loss.

    Focal Loss is designed to address the problem of class imbalance in object detection.

    Args:
        output (torch.Tensor): The predicted output from the model.
        target (torch.Tensor): The true labels.

    Returns:
        torch.Tensor: The computed focal loss.
    """
    ce_loss = F.cross_entropy(output, target, reduction='none')
    #print(ce_loss.shape)
    pt = torch.exp(-ce_loss)
    alpha = 0.25
    gamma = 2
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
    return focal_loss