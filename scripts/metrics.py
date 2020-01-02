import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


def soft_binary_cross_entropy(pred, soft_targets):
    L = -torch.sum((soft_targets * torch.log(nn.functional.sigmoid(pred)) +
                    (1. - soft_targets) * torch.log(nn.functional.sigmoid(1. - pred))), 1)
    return torch.mean(L)


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)
