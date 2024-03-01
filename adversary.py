import numpy as np
import torch
import torch.nn as nn


class Attack:
    def __init__(self, net, criterion):
        # 网络参数
        self.net = net
        # 交叉熵损失函数
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, epsilon=0.03, x_val_max=-100, x_val_min=100):
        x_adv = torch.tensor(x.data, requires_grad=True)
        h_adv = self.net(x_adv)
        if targeted:        # 靶向攻击, 此时的y是攻击的目标
            cost = self.criterion(h_adv, y.long())
        else:               # 非靶向攻击, 此时的y是x对应的真实值
            cost = -self.criterion(h_adv, y.long())
            print("epsilon: ", epsilon)
        # 梯度清零
        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.zero.fill_(0)
        cost.backward()

        # print(x_adv.grad)
        # print(x_adv.grad.sign())
        print(x_adv.min().abs(), x_adv.max().abs())
        # x的梯度取符号
        # x_adv_grad = x_adv.grad.sign()*torch.max(x_adv)
        # x_adv_grad = x_adv.grad
        # x_adv_grad = x_adv.grad.sign() * torch.max(x_adv)
        x_adv_grad = x_adv.grad.sign() * max(x_adv.min().abs(), x_adv.max().abs())
        x_adv = x_adv - epsilon*x_adv_grad

        # 将x_adv的值限制在 x_val_min 到 x_val_max 之间
        # x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        h = self.net(x_adv)
        print((np.array(x.detach().cpu())==np.array(x_adv.detach().cpu())).sum()/len(x.detach().cpu().numpy().flatten()))
        return x_adv, h_adv, h
    def ifgsm(self):
        pass

    def PGD(self):
        pass