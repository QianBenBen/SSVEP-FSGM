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
from adversary import *
from utils import *


class Solution:
    def __init__(self, args):
        # 参数
        self.args = args
        # 是否使用gpu
        self.cuda = (args.cuda and torch.cuda.is_available())
        # 设备
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")
        # 训练轮次
        self.epoch = args.epoch
        # 训练批次大小
        self.batch_size = args.batch_size
        # 学习率
        self.lr = args.lr
        # 分类的类别总数
        self.nb_classes = args.nb_classes
        # eeg数据的通道数
        self.channels = args.channels
        # 采样率
        self.samp_rate = args.samp_rate
        # 样本采样点数
        self.sample_length = args.samples
        # checkpoint名
        self.checkpoint_name = args.checkpoint

        # 攻击目标
        self.target = args.target
        # ifgsm的攻击迭代次数
        self.iteration = args.iteration
        # fgsm的扰动幅度
        self.epsilon = args.epsilon
        #
        self.alpha = args.alpha

        # 数据集名称
        self.dataset_name = args.dataset
        # 训练迭代次数统计
        self.global_epoch = 0
        # 训练iter次数
        self.global_iter = 0

        # 程序运行模式
        self.mode = args.mode
        # 环境名称
        self.env_name = args.env_name
        # 网络模型名称
        self.model_name = args.model


        # 创建模型保存所在文件夹
        # self.user = input("输入用户姓名: ")
        self.ckpt_dir = os.path.join(args.ckpt_dir, self.env_name + "/" + self.model_name)
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

        # 数据加载
        self.load_data()
        # 网络模型初始化
        self.model_init()
        # 交叉熵函数
        self.criterion = nn.CrossEntropyLoss()

        # Histories
        self.history = dict()
        self.history['acc'] = 0.
        self.history['epoch'] = 0
        self.history['iter'] = 0

        # 模型载入
        if self.mode!="train" and args.checkpoint != '':
            self.load_checkpoint(args.checkpoint)

        # 攻击算法
        self.attack_method = Attack(self.net, self.criterion)


    def train(self):
        # 训练模式
        self.net.train()
        # Training Loop
        start_time = time.time()
        for epoch in range(self.epoch):
            self.global_epoch+=1
            print(f"---------- EPOCH {epoch + 1} ----------")
            for inputs, labels in self.train_loader:
                self.global_iter += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 优化器清除梯度
                self.optim.zero_grad()
                outputs = self.net(inputs)
                # 交叉熵计算损失
                loss = self.criterion(outputs, labels.long())
                # 预测结果
                predict = torch.argmax(outputs, dim=1)
                acc = torch.eq(predict, labels).float().mean()
                # 优化器优化模型
                loss.backward()
                self.optim.step()
                # 误差分析
                if self.global_iter % 100 == 0:
                    end_time = time.time()
                    print(f"{end_time - start_time:.2f}\t 训练次数：{self.global_iter}, acc：{acc}, Loss：{loss}")
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
            self.save_checkpoint(self.checkpoint_name)

    def attack(self, num_sample=100, target=-1, epsilon=0.03, alpha=2/255, iteration=1):
        # 设为非训练模式
        self.net.eval()
        sample_x, sample_y = self.sample_data(100)
        if isinstance(target, int) and (target in range(self.nb_classes)):
            y_target = torch.LongTensor(sample_y.size()).fill_(target)
        else:
            y_target = None

        x_adv, (accuracy, cost, accuracy_adv, cost_adv) = self.FGSM(sample_x, sample_y, y_target, epsilon, self.alpha, self.iteration)

        print('[BEFORE] accuracy : {:.2f} cost : {:.3f}'.format(accuracy, cost))
        print('[AFTER] accuracy : {:.2f} cost : {:.3f}'.format(accuracy_adv, cost_adv))



    def FGSM(self, x, y_true, y_target=None, eps=0.03, alpha=2/255, iteration=1):
        x = x.to(self.device)
        x.requires_grad = True
        y_true = y_true.to(self.device)

        # 靶向攻击 或 非靶向攻击
        if y_target is not None:
            targeted_attack = True
            y_target = y_target.to(self.device)
        else:
            targeted_attack = False

        # 模型对样本置信度预测
        h = self.net(x)
        # 模型对样本的预测结果
        predict = torch.argmax(h, dim=1)
        # 模型对样本的预测准确度
        accuracy = torch.eq(predict, y_true).float().mean()
        # 计算交叉熵损失值
        cost = self.criterion(h, y_true.long())

        # aa = x.is_leaf
        # bb = y_true.is_leaf
        if iteration==1:
            if targeted_attack==True:
                x_adv, h_adv, h = self.attack_method.fgsm(x, y_target, True, eps)
            else:
                x_adv, h_adv, h = self.attack_method.fgsm(x, y_true, False, eps)
        else:
            if targeted_attack==True:
                x_adv, h_adv, h = self.attack_method.i_fgsm(x, y_target, True, eps)
            else:
                x_adv, h_adv, h = self.attack_method.i_fgsm(x, y_true, False, eps)

        predict_adv = torch.argmax(h_adv, dim=1)
        # predict_h_after = torch.argmax(h, dim=1)
        # print(predict)
        # print(predict_adv)
        accuracy_adv = torch.eq(predict_adv, y_true).float().mean()
        # accuracy_h_after = torch.eq(predict_h_after, y_true).float().mean()
        cost_adv = self.criterion(h_adv, y_true.long())

        print("True label: ", y_true.clone().cpu().numpy().tolist())
        print("Predict: ", predict.clone().cpu().numpy().tolist())
        print("predict_adv: ", predict_adv.clone().cpu().numpy().tolist())

        return x_adv, (accuracy, cost, accuracy_adv, cost_adv)


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
        # print("模型权重加载成功")

    def load_data(self):
        load_data_mode = 1
        # print(f"self.checkpoint_name[-5] : {self.checkpoint_name[-5]}")
        if self.checkpoint_name[-5] == "2":
            load_data_mode = 2
        if self.dataset_name == "Benchmark":
            train_data = Benchmark("E:\Datasets\BCI\SSVEP\Benchmark", train=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]), mode=load_data_mode)
            test_data = Benchmark("E:\Datasets\BCI\SSVEP\Benchmark", train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]), mode=load_data_mode)
            # dataloader
            self.train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            self.test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        elif self.dataset_name == "Beta":
            pass
        else:
            raise UnknownDatasetError()

    def sample_data(self, num_sample=100):
        total = len(self.test_loader.dataset)
        # 从全体样本中选取num_sample个样本
        seed = torch.tensor(np.random.randint(0, total, size=num_sample))

        x = self.test_loader.dataset.data[seed]
        # x = self.scale(x.float().unsqueeze(1).div(255))
        y = self.test_loader.dataset.label[seed]
        x = torch.tensor(x)
        x = x.float()
        y = torch.tensor(y)

        return x, y




