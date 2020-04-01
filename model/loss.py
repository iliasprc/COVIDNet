import torch.nn.functional as F

import torch
def nll_loss(output, target):
    return F.nll_loss(output, target)


def crossentropy_loss(output, target):
    return F.cross_entropy(output, target)


def focal_loss(output,target):
    ce_loss = F.cross_entropy(output, target, reduction='none')
    #print(ce_loss.shape)
    pt = torch.exp(-ce_loss)
    alpha = 0.25
    gamma = 2
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
    return focal_loss