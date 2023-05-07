import torch

def weighted_smooth_l1_loss(input, target, weights):
    batch_loss = (torch.abs(input - target)<1).float()*(input - target)**2 +\
        (torch.abs(input - target)>=1).float()*(torch.abs(input - target) - 0.5)
    weighted_batch_loss = weights * batch_loss 
    weighted_loss = weighted_batch_loss.sum()
    return weighted_loss