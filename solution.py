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
from datasets import *
from utils import *

class Solution:
    def __init__(self, args):
        self.args = args
        self.cuda = (args.cuda and torch.cuda.is_available())
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.nb_classes = args.nb_classes
        self.channels = args.channels
        self.target = args.target
        self.samp_rate = args.samp_rate
        self.sample_length = args.samples

        self.iteration = args.iteration
        self.epsilon = args.epsilon
        self.alpha = args.alpha

        self.dataset_name = args.dataset
        self.global_epoch = 0
        self.global_iter = 0

        self.mode = args.mode
        self.env_name = args.env_name
        self.model_name = args.model

        # self.user = input("输入用户姓名: ")
        self.ckpt_dir = os.path.join(args.ckpt_dir, self.env_name + "/" + self.model_name)
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

        self.load_data()
        self.model_init()
        self.criterion = nn.CrossEntropyLoss()
        # Histories
        self.history = dict()
        self.history['acc'] = 0.
        self.history['epoch'] = 0
        self.history['iter'] = 0

        if args.checkpoint != '':
            self.load_checkpoint(args.checkpoint)

    def train(self):
        # 训练模式
        self.net.train()
        # Training Loop
        start_time = time.time()
        for epoch in range(self.epoch):
            print(f"---------- EPOCH {epoch + 1} ----------")
            for inputs, labels in self.train_loader:
                self.global_iter += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 优化器清除梯度
                self.optim.zero_grad()
                outputs = self.net(inputs)
                # 交叉熵计算损失
                loss = self.criterion(outputs, labels.long())
                #
                predict = torch.argmax(outputs, dim=1)
                acc = torch.eq(predict, labels).float().mean()
                # 优化器优化模型
                loss.backward()
                self.optim.step()
                # 误差分析
                if self.global_iter % 100 == 0:
                    end_time = time.time()
                    print(f"{end_time - start_time}\t 训练次数：{self.global_iter}, acc：{acc}, Loss：{loss}")
                    # writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 测试
        self.test()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                values, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_samples += len(labels)

            accuracy = total_correct / total_samples
            print(f"Test Accuracy: {accuracy:.4f}")

        if self.history['acc'] < accuracy:
            self.history['acc'] = accuracy
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            self.save_checkpoint('best_acc.tar')

    def model_init(self):
        # 必须要可以访问倒EEGNet结构
        # select model type
        if self.model_name == "EEGNet":
            self.net = EEGNet(self.nb_classes, self.channels, self.sample_length)
        elif self.model_name == "DeepCNN":
            pass
        elif self.model_name == "ShallowCNN":
            pass
        else:
            raise ValueError("目标模型不存在!模型须在 EEGNet / DeepCNN / ShallowCNN 中选择")

        self.net = self.net.to(self.device)
        self.net.weight_init()
        self.optim = torch.optim.Adam(params=self.net.parameters(), lr=self.lr)


    def save_checkpoint(self, filename):
        model_state = self.net.state_dict()
        optim_state = self.optim.state_dict()
        state = {
            "model_state": model_state,
            "optim_state": optim_state,
            "lr": self.lr,
            "epoch": self.global_epoch,
            "iter": self.global_iter,
            "history": self.history,
            "dataset": self.dataset_name,
            "args": self.args
        }
        filepath = self.ckpt_dir + "/" + filename
        torch.save(state, filepath)
        print("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))


    def load_checkpoint(self, filename):
        filepath = self.ckpt_dir + "/" + filename
        try:
            state = torch.load(filepath)
        except IOError:
            print("未找到权重文件")
            raise
        else:
            print(f"权重文件 {filename} 载入成功")
        self.net.load_state_dict(state['model_state'])
        self.optim.load_state_dict(state['optim_state'])
        self.global_epoch = state['epoch']
        self.global_iter = state['iter']
        print("模型权重加载成功")

    def load_data(self):
        if self.dataset_name == "Benchmark":
            train_data = Benchmark("E:\Datasets\BCI\SSVEP\Benchmark", train=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]))
            test_data = Benchmark("E:\Datasets\BCI\SSVEP\Benchmark", train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))
            # dataloader
            self.train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            self.test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        elif self.dataset_name == "Beta":
            pass
        else:
            raise UnknownDatasetError()




