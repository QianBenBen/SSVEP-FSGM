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


def loss_compute(batch_size, confidence_score, target_label, k=40, num_labels=1000, targeted=True):
    # print(f"计算一个batch中{batch_size}张图片的loss")
    target_score, _ = torch.max(confidence_score * target_label, dim=1)
    other_label = (1 - target_label)
    other_score, _ = torch.max(other_label * confidence_score, dim=1)
    if targeted:
        ll = other_score - target_score + k
    else:
        ll = target_score - other_score + k
    ll = torch.clamp(ll, min=0)  # 限制范围，截取功能
    return ll
