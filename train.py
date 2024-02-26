from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.datasets import MNIST
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import time

from model import EEGNet
from datasets import Benchmark


def eeg_train(learning_rate: float = 1e-3,
              nb_classes: int = 40,
              Chans: int = 9,
              Samples: int = 1375,
              num_epochs: int = 200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.set_default_dtype(torch.float64)
    train_data = Benchmark("E:\Datasets\BCI\SSVEP\Benchmark", train=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
    test_data = Benchmark("E:\Datasets\BCI\SSVEP\Benchmark", train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                          ]))
    print("train_data:\n", len(train_data))
    print(f"shape: {train_data[0][0].shape}, type: {type(train_data[0][0])}")
    print("test_data:\n", len(test_data))
    print(f"shape: {test_data[0][0].shape}, type: {type(test_data[0][0])}")

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # tensorboard 记录训练结果
    writer = SummaryWriter("./logs_train")

    # dataloader
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    eegnet = EEGNet(nb_classes, Chans, Samples)
    eegnet = eegnet.to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(eegnet.parameters(), lr=learning_rate, foreach=False)

    # Training Loop
    start_time = time.time()
    total_train_step = 0
    for epoch in range(num_epochs):
        eegnet.train()
        print(f"---------- EPOCH {epoch + 1} ----------\n")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 优化器清除梯度
            optimizer.zero_grad()
            outputs = eegnet(inputs)
            # 交叉熵计算损失
            loss = criterion(outputs, labels.long())
            # 优化器优化模型
            loss.backward()
            optimizer.step()
            # 误差分析
            total_train_step += 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print(f"{end_time - start_time}\t 训练次数：{total_train_step}, Loss：{loss}")
                writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    eegnet.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = eegnet(inputs)
            values, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += len(labels)

        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")

    torch.save(eegnet, f".\Weights/eeg_gpu.pth")
    # torch.save(light.state_dict(), f"Weights/light_{epoch}.pth")
    print("模型已保存")
    writer.close()