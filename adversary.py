import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from utils.utils import *

class Attack:
    def __init__(self, net, criterion, start_time):
        # 网络参数
        self.net = net
        # 交叉熵损失函数
        self.criterion = criterion
        # 开始时间
        self.start_time = start_time

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

    # 单样本CW攻击
    def CW(self, input: torch.Tensor,
           label: int,
           num_labels: int = 1000,
           targeted: bool = False,
           confidence: float = 0,
           learning_rate: float = 0.01,
           initial_const: float = 0.001,
           binary_search_steps: int = 9,
           max_iterations: int = 1000,
           boxmin: float = -1.0,
           boxmax: float = 1.0,
           abort_early: bool = True) -> torch.Tensor:
        """
        Carlini and Wagner L2 attack from https://arxiv.org/abs/1608.04644.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : Tensor
            Inputs to attack. Should be in [0, 1].
        labels : Tensor
            Labels corresponding to the inputs if untargeted, else target labels.
        targeted : bool
            Whether to perform a targeted attack or not.
        confidence : float
            Confidence of adversarial examples: higher produces examples that are farther away, but more strongly classified
            as adversarial.
        learning_rate: float
            The learning rate for the attack algorithm. Smaller values produce better results but are slower to converge.
        initial_const : float
            The initial tradeoff-constant to use to tune the relative importance of distance and confidence. If
            binary_search_steps is large, the initial constant is not important.
        binary_search_steps : int
            The number of times we perform binary search to find the optimal tradeoff-constant between distance and
            confidence.
        max_iterations : int
            The maximum number of iterations. Larger values are more accurate; setting too small will require a large, 1000次可以完成95%的优化工作
            learning rate and will produce poor results.
        boxmin : float
            像素值下界
        boxmax : float
            像素值上界
        abort_early : bool
            If true, allows early aborts if gradient descent gets stuck.
        callback : Optional

        Returns
        -------
        adv_inputs : Tensor
            Modified inputs to be adversarial to the model.

        """
        device = input.device
        k = 40  # k值
        # 攻击目标标签，使用one hot编码更利于计算
        # target_label 的 onehot 编码方式
        tlab = torch.tensor(np.eye(num_labels)[label]).to(device).float()

        # c的上下界
        lower_bound = 0
        upper_bound = 1e10
        # c的初始化边界
        c = initial_const
        # 是否攻击成功
        success = False

        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = [torch.zeros(size=input.shape)]

        # the resulting image, tanh'd to keep bound from boxmin to boxmax
        boxmul = (boxmax - boxmin) / 2.0
        boxplus = (boxmin + boxmax) / 2.0

        prev = float("inf")

        for outer_step in range(binary_search_steps):
            # print("o_bestl2={} c={}".format(o_bestl2, c))

            # 把原始图像转换成图像数据和扰动的形态
            timg = torch.arctanh((input - boxplus) / boxmul * 0.99999).clone().detach().to(device).float()
            modifier = torch.zeros_like(timg).to(device).float()
            # 图像数据的扰动量梯度可以获取
            modifier.requires_grad = True
            optimizer = torch.optim.Adam([modifier], lr=learning_rate)  # 优化器
            for iteration in range(1, max_iterations + 1):
                optimizer.zero_grad()
                # 定义新输入
                newimg = torch.tanh(modifier + timg) * boxmul + boxplus
                output = self.net(newimg)
                # 定义cw中的损失函数  l2范数。torch.dist指计算两个张量的距离
                loss2 = torch.dist(newimg, (torch.tanh(timg) * boxmul + boxplus), p=2)

                real = torch.max(output * tlab)
                other = torch.max((1 - tlab) * output)
                if targeted:
                    loss1 = other - real + k
                else:
                    loss1 = real - other + k
                loss1 = torch.clamp(loss1, min=0)  # 限制范围，截取功能
                loss1 = c * loss1

                loss = loss1 + loss2

                loss.backward(retain_graph=True)
                optimizer.step()
                l2 = loss2
                sc = output.data.cpu()

                # 当前c值太糟糕情况下尽快停止搜索。check if we should abort search if we're getting nowhere.
                if abort_early and iteration % (max_iterations // 10) == 0:
                    if (loss > prev * 0.9999).all():
                        break
                    prev = loss.detach()

                # # 隔段显示信息
                # if (iteration % (max_iterations // 10) == 0):
                #     print("iteration={} loss={} loss1={} loss2={}".format(
                #         iteration, loss, loss1, loss2))

                # 保存损失函数最小，并且攻击成功的对抗样本
                if (l2 < o_bestl2) and (torch.argmax(sc) == label):
                    print(f"{time.time()-self.start_time:.2f}  attack l2={l2} target_label={label}")
                    o_bestl2 = l2
                    o_bestscore = torch.argmax(sc)
                    o_bestattack = newimg.data.cpu()

            c_old = -1
            if (o_bestscore == label) and o_bestscore != -1:
                success = True
                # 攻击成功，减小c
                upper_bound = min(upper_bound, c)
                if upper_bound < 1e9:
                    print(f"{time.time()-self.start_time:.2f}  Binary searching c")
                    c_old = c
                    c = (lower_bound + upper_bound) / 2
            else:
                # 攻击失败，增加c
                lower_bound = max(lower_bound, c)
                c_old = c
                if upper_bound < 1e9:
                    c = (lower_bound + upper_bound) / 2
                else:
                    c *= 10
        print(f"c {initial_const}->{c}")
        return (success, o_bestscore, o_bestl2, o_bestattack)


    def carlini_wagner_l2(self, inputs: torch.Tensor,
                          labels: torch.Tensor,
                          label_num: int = 1000,
                          targeted: bool = False,
                          confidence: float = 0,
                          learning_rate: float = 0.01,
                          initial_const: float = 0.001,
                          binary_search_steps: int = 9,
                          max_iterations: int = 1000,
                          abort_early: bool = True) -> torch.Tensor:
        """
        Carlini and Wagner L2 attack from https://arxiv.org/abs/1608.04644.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : Tensor
            Inputs to attack. Should be in [0, 1].
        labels : Tensor
            Labels corresponding to the inputs if untargeted, else target labels.
        targeted : bool
            Whether to perform a targeted attack or not.
        confidence : float
            Confidence of adversarial examples: higher produces examples that are farther away, but more strongly classified
            as adversarial.
        learning_rate: float
            The learning rate for the attack algorithm. Smaller values produce better results but are slower to converge.
        initial_const : float
            The initial tradeoff-constant to use to tune the relative importance of distance and confidence. If
            binary_search_steps is large, the initial constant is not important.
        binary_search_steps : int
            The number of times we perform binary search to find the optimal tradeoff-constant between distance and
            confidence.
        max_iterations : int
            The maximum number of iterations. Larger values are more accurate; setting too small will require a large
            learning rate and will produce poor results.
        abort_early : bool
            If true, allows early aborts if gradient descent gets stuck.
        callback : Optional

        Returns
        -------
        adv_inputs : Tensor
            Modified inputs to be adversarial to the model.

        """
        boxmin = -1.0
        boxmax = 1.0
        boxmul = (boxmax - boxmin) / 2.0
        boxplus = (boxmin + boxmax) / 2.0

        device = inputs.device
        batch_size = len(inputs)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
        # (tanh(t_inputs)+1)/2=inputs 反过来 ： 是因为计算梯度的时候，子元素是 t_inputs 吗？ 所以加的扰动也需要与 t_inputs 相加
        t_inputs = torch.arctanh((inputs - boxplus) / boxmul * (1 - 1e-5)).to(device)
        # 靶向则 multiplier 为 -1, 否则为 1
        multiplier = -1 if targeted else 1

        # set the lower and upper bounds of c accordingly
        c = torch.full((batch_size,), initial_const, device=device).float()
        lower_bound = torch.zeros_like(c)
        upper_bound = torch.full_like(c, 1e10)

        # 将标签初始化为 onehot/inf_hot 编码形式
        # 严格的单标签分类问题编码形式: [0, 0, 1]
        labels_onehot = torch.zeros((batch_size, label_num)).to(device).scatter_(1, labels.unsqueeze(1), 1)
        # 多标签分类问题编码形式，标明各个类别的概率: [0.2, 0.3, 0.5]
        # labels_infhot = torch.zeros((batch_size, label_num)).to(device).scatter_(1, labels.unsqueeze(1), float('inf'))

        '''设置最优参数'''
        # 当前batchsize l2范数下最优c列表
        o_best_l2 = torch.full_like(c, float('inf')).to(device)
        # 当前batchsize 最优对抗样本列表
        o_best_adv = inputs.clone().to(device)
        # 当前batchsize 是否找到对抗样本列表
        o_adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)

        i_total = 0
        for outer_step in range(binary_search_steps):

            # setup the modifier variable and the optimizer
            modifier = torch.zeros_like(inputs, requires_grad=True)
            # 为什么优化器的输入参数是 [modifier]: 输入的参数是优化目标
            optimizer = optim.Adam([modifier], lr=learning_rate)
            # 当前 c 值下最好的 l2范数 距离
            best_l2 = torch.full_like(c, float('inf'))
            adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)

            # 若2分查找次数大于等于10 并且 当前为最后一次查找，c自动变为上界.
            if (binary_search_steps >= 10) and outer_step == (binary_search_steps - 1):
                c = upper_bound

            # prev设置为浮点类型的无穷大
            prev = float('inf')
            for i in range(max_iterations):
                optimizer.zero_grad()

                # 对抗样本的输入，扰动直接与t_inputs相加
                adv_inputs = torch.tanh(t_inputs + modifier) * boxmul + boxplus
                # l2范数的计算
                l2_squared = (adv_inputs - inputs).flatten(1).square().sum(1)
                l2 = l2_squared.detach().sqrt()
                # 模型预测结果
                logits = self.net(adv_inputs)

                # 计算总优化目标
                sub_loss = loss_compute(batch_size, logits, labels_onehot, k=40, num_labels=label_num,
                                        targeted=targeted)
                loss = l2 + c * sub_loss

                # adjust the best result found so far
                # 为了提高预测的置信度，我个人的理解是没必要，论文里面也没说
                # predicted_classes = (logits - labels_onehot * confidence).argmax(1) if targeted else \
                #     (logits + labels_onehot * confidence).argmax(1)
                predicted_classes = logits.argmax(1)

                # 当前c值数据统计
                # 是否攻击成功
                is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
                # 是否当前对抗样本的l2范数更小
                is_smaller = l2 < best_l2
                # 是否又是对抗样本，l2范数又最小
                is_both = is_adv & is_smaller
                best_l2 = torch.where(is_both, l2, best_l2)
                adv_found.logical_or_(is_both)

                # 全局数据统计
                o_is_smaller = l2 < o_best_l2
                o_is_both = is_adv & o_is_smaller
                o_best_l2 = torch.where(o_is_both, l2, o_best_l2)
                o_adv_found.logical_or_(is_both)
                o_best_adv = torch.where(batch_view(o_is_both), adv_inputs.detach(), o_best_adv)

                # 当前c值太糟糕情况下尽快停止搜索。check if we should abort search if we're getting nowhere.
                if abort_early and i % (max_iterations // 10) == 0:
                    if (loss > prev * 0.9999).all():
                        break
                    prev = loss.detach()

                loss.backward(retain_graph=True)
                # modifier.grad = grad(loss.sum(), modifier, only_inputs=True)[0]
                optimizer.step()

            old_c = c.detach().clone()

            # 改变各个c的值。adjust the constant as needed
            # 攻击成功 则 将c上界修改为当前c值
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], c[adv_found])
            adv_not_found = ~adv_found
            # 攻击失败 则 将c下界修改为当前c值
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], c[adv_not_found])
            # 上界小于 1e9 序列
            is_smaller = upper_bound < 1e9
            # 二分查找 c 的值
            c[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            # 若上界大于 1e9 且 攻击失败， 则将 c的值扩大至10倍
            c[(~is_smaller) & adv_not_found] *= 10

            print(f"outer_step={outer_step} c: {old_c}->\n{c}\nAttack condition: {adv_found}\n\n")
        print(f"Predict: {torch.argmax(self.net(o_best_adv), dim=1)}")
        # return the best solution found
        return o_best_l2, o_best_adv

    def PGD(self):
        pass