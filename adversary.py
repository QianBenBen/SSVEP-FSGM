import numpy as np
import torch
import torch.nn as nn

from utils.utils import *

class Attack:
    def __init__(self, net, criterion):
        # 网络参数
        self.net = net
        # 交叉熵损失函数
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, epsilon=0.03, x_val_max=-1, x_val_min=1):
        print("Attack with fgsm")
        abs_max = max(x.min().abs(), x.max().abs())
        x = x / abs_max

        # 新建一个数据相同但是不共享计算图的tensor
        x_adv = x.clone().detach()
        x_adv.requires_grad=True
        # x_adv = torch.tensor(x.data, requires_grad=True)

        x_adv.retain_grad()
        h_adv = self.net(x_adv)
        if targeted:        # 靶向攻击, 此时的y是攻击的目标
            cost = self.criterion(h_adv, y.long())
            print("targeted attack: ", y[0])
        else:               # 非靶向攻击, 此时的y是x对应的真实值
            cost = -self.criterion(h_adv, y.long())
            print("nontargeted attack ")

        # 梯度清零
        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.zero.fill_(0)
        cost.backward()

        x_grad_sign = x_adv.grad.sign()
        x_adv = x_adv - epsilon*x_grad_sign

        # print(f"average change percent:{(x_adv-x).abs().mean()}")
        # 将x_adv的值限制在 x_val_min 到 x_val_max 之间
        # x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        x = x * abs_max
        x_adv = x_adv * abs_max

        print(f"max:{abs_max}, average change:{(x_adv-x).abs().mean()}")

        h = self.net(x)
        h_adv = self.net(x_adv)
        # print((np.array(x.detach().cpu())==np.array(x_adv.detach().cpu())).sum()/len(x.detach().cpu().numpy().flatten()))
        return x_adv, h_adv, h

    def i_fgsm(self, x, y, targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min=-1, x_val_max=1):
        print("Attack with i_fgsm")
        abs_max = max(x.min().abs(), x.max().abs())
        x = x / abs_max

        # 新建一个数据相同但是不共享计算图的tensor
        x_adv = x.clone().detach()
        x_adv.requires_grad = True
        # x_adv = torch.tensor(x.data, requires_grad=True)

        for i in range(iteration):
            h_adv = self.net(x_adv)
            if targeted:
                cost = self.criterion(h_adv, y.long())
                print("targeted attack: ", y[0])
            else:
                cost = -self.criterion(h_adv, y.long())
                print("nontargeted attack ")
            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv_grad = x_adv.grad.sign()
            x_adv = x_adv - alpha*x_adv_grad
            x_adv = where(x_adv > x+eps, x+eps, x_adv)
            x_adv = where(x_adv < x-eps, x-eps, x_adv)
            # x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = x_adv.clone().detach()
            x_adv.requires_grad = True

        x = x * abs_max
        x_adv = x_adv * abs_max

        h = self.net(x)
        h_adv = self.net(x_adv)

        print(f"max:{abs_max}, average change:{(x_adv-x).abs().mean()}")
        return x_adv, h_adv, h

    def PGD(self):
        pass