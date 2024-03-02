import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import Benchmark
from model import EEGNet, DepthwiseSeparableConv2d


def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)
