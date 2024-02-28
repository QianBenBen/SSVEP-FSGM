import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import Benchmark
from model import EEGNet, DepthwiseSeparableConv2d

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

