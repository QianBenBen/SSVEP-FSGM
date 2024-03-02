from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy
from scipy import signal
from tqdm import tqdm

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

class Benchmark(Dataset):
    classes = {

    }

    stim_event_freq = [8., 8.2, 8.4, 8.6, 8.8, 9., 9.2, 9.4, 9.6, 9.8, 10., 10.2, 10.4, 10.6,
                       10.8, 11., 11.2, 11.4, 11.6, 11.8, 12., 12.2, 12.4, 12.6, 12.8, 13., 13.2, 13.4,
                       13.6, 13.8, 14., 14.2, 14.4, 14.6, 14.8, 15., 15.2, 15.4, 15.6, 15.8]

    def __init__(
            self,
            root: str = '',
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            mode: int = 1
    ) -> None:
        super(Dataset).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.subject_num = 35
        # 采样率1000，降采样至 250Hz
        self.samp_rate = 250
        # 预处理滤波器设置
        '''没看懂'''
        self.filterB, self.filterA = self.__get_pre_filter(self.samp_rate)
        self.mode = mode
        if self.mode == 1:
            self.data, self.pre_data, self.label = self.load_data()
        else:
            self.data, self.pre_data, self.label = self.load_data2()

    # 用每个被试的前5个block
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        channels = [53, 54, 55, 57, 58, 59, 61, 62, 63]
        channels = [i - 1 for i in channels]

        if self.train:
            # train data
            print("---------- 训练数据加载 ----------", flush=False)
            data = np.zeros((200 * self.subject_num, 1, len(channels), 1375))
            pre_data = np.zeros((200 * self.subject_num, 1, len(channels), 125))
            label = np.zeros(200 * self.subject_num, dtype=int)
        else:
            # test data
            print("---------- 测试数据加载 ----------", flush=False)
            data = np.zeros((40 * self.subject_num, 1, len(channels), 1375))
            pre_data = np.zeros((40 * self.subject_num, 1, len(channels), 125))
            label = np.zeros(40 * self.subject_num, dtype=int)

        for sub_num in tqdm(range(1, self.subject_num + 1)):
            f = scipy.io.loadmat(self.root + f"/S{sub_num}.mat")
            # print(f"mat{sub_num}文件大小: {f['data'].shape}")
            for block in range(6):
                for target in range(40):
                    if self.train and block != 5:
                        data[(sub_num - 1) * 200 + block * 40 + target][0] = f["data"][channels, 125:, target, block]
                        pre_data[(sub_num - 1) * 200 + block * 40 + target][0] = f["data"][channels, :125, target, block]
                        label[(sub_num - 1) * 200 + block * 40 + target] = int(target)
                    elif not self.train and block == 5:
                        data[(sub_num - 1) * 40 + target][0] = f["data"][channels, 125:, target, block]
                        pre_data[(sub_num - 1) * 40 + target][0] = f["data"][channels, :125, target, block]
                        label[(sub_num - 1) * 40 + target] = int(target)
        return data, pre_data, label

    def load_data2(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        channels = [53, 54, 55, 57, 58, 59, 61, 62, 63]
        channels = [i - 1 for i in channels]

        if self.train:
            # train data
            print("---------- 训练数据加载 ----------", flush=False)
            data = np.zeros((int(240 * self.subject_num * 0.8), 1, len(channels), 1375))
            pre_data = np.zeros((int(240 * self.subject_num * 0.8), 1, len(channels), 125))
            label = np.zeros(int(240 * self.subject_num * 0.8), dtype=int)
        else:
            # test data
            print("---------- 测试数据加载 ----------", flush=False)
            data = np.zeros((int(240 * self.subject_num * 0.2), 1, len(channels), 1375))
            pre_data = np.zeros((int(240 * self.subject_num * 0.2), 1, len(channels), 125))
            label = np.zeros(int(240 * self.subject_num * 0.2), dtype=int)

        start_subject = 1 if self.train else 29
        end_subject = 29 if self.train else 36
        for sub_num in tqdm(range(start_subject, end_subject)):
            f = scipy.io.loadmat(self.root + f"/S{sub_num}.mat")
            # print(f"mat{sub_num}文件大小: {f['data'].shape}")
            for block in range(6):
                for target in range(40):
                    data[(sub_num - start_subject) * 240 + block * 40 + target][0] = f["data"][channels, 125:, target, block]
                    pre_data[(sub_num - start_subject) * 240 + block * 40 + target][0] = f["data"][channels, :125, target,
                                                                          block]
                    label[(sub_num - start_subject) * 240 + block * 40 + target] = int(target)
        return data, pre_data, label

    def __get_pre_filter(self, samp_rate):
        fs = samp_rate
        f0 = 50
        q = 35
        b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
        return b, a

    def __preprocess(self, data):
        filter_data = signal.filtfilt(self.filterB, self.filterA, data)
        return filter_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        eeg, target = self.data[index], self.label[index]

        # 滤波处理
        eeg = self.__preprocess(eeg)

        if self.transform is not None:
            eeg = self.transform(eeg.copy())
            if eeg.dim()==3:
                eeg = torch.permute(eeg, dims=(1, 2, 0))

        if self.target_transform is not None:
            target = self.target_transform(target.copy())

        eeg = eeg.float()
        return eeg, target





    class Beta(Dataset):
        '''
        Beta数据集未完成，请勿使用
        '''
        classes = {

        }

        stim_event_freq = [8., 8.2, 8.4, 8.6, 8.8, 9., 9.2, 9.4, 9.6, 9.8, 10., 10.2, 10.4, 10.6,
                           10.8, 11., 11.2, 11.4, 11.6, 11.8, 12., 12.2, 12.4, 12.6, 12.8, 13., 13.2, 13.4,
                           13.6, 13.8, 14., 14.2, 14.4, 14.6, 14.8, 15., 15.2, 15.4, 15.6, 15.8]

        def __init__(
                self,
                root: str = '',
                train: bool = True,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
        ) -> None:
            super(Dataset).__init__()
            self.root = root
            self.train = train
            self.transform = transform
            self.target_transform = target_transform
            self.subject_num = 35
            # 采样率1000，降采样至 250Hz
            self.samp_rate = 250
            # 预处理滤波器设置
            '''没看懂'''
            self.filterB, self.filterA = self.__get_pre_filter(self.samp_rate)
            self.data, self.pre_data, self.label = self.load_data()

        # 用每个被试的前5个block
        def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            channels = [53, 54, 55, 57, 58, 59, 61, 62, 63]
            channels = [i - 1 for i in channels]

            if self.train:
                # train data
                print("---------- 训练数据加载 ----------", flush=False)
                data = np.zeros((200 * self.subject_num, len(channels), 1375))
                pre_data = np.zeros((200 * self.subject_num, len(channels), 125))
                label = np.zeros(200 * self.subject_num, dtype=int)
            else:
                # test data
                print("---------- 测试数据加载 ----------", flush=False)
                data = np.zeros((40 * self.subject_num, len(channels), 1375))
                pre_data = np.zeros((40 * self.subject_num, len(channels), 125))
                label = np.zeros(40 * self.subject_num, dtype=int)

            for sub_num in tqdm(range(1, self.subject_num + 1)):
                f = scipy.io.loadmat(self.root + f"/S{sub_num}.mat")
                # print(f"mat{sub_num}文件大小: {f['data'].shape}")
                for block in range(6):
                    for target in range(40):
                        if self.train and block != 5:
                            data[(sub_num - 1) * 200 + block * 40 + target] = f["data"][channels, 125:, target, block]
                            pre_data[(sub_num - 1) * 200 + block * 40 + target] = f["data"][channels, :125, target,
                                                                                  block]
                            label[(sub_num - 1) * 200 + block * 40 + target] = int(target)
                        elif not self.train and block == 5:
                            data[(sub_num - 1) * 40 + target] = f["data"][channels, 125:, target, block]
                            pre_data[(sub_num - 1) * 40 + target] = f["data"][channels, :125, target, block]
                            label[(sub_num - 1) * 40 + target] = int(target)
            return data, pre_data, label

        def load_data2(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            channels = [53, 54, 55, 57, 58, 59, 61, 62, 63]
            channels = [i - 1 for i in channels]

            if self.train:
                # train data
                print("---------- 训练数据加载 ----------", flush=False)
                data = np.zeros((int(240 * self.subject_num * 0.8), 1, len(channels), 1375))
                pre_data = np.zeros((int(240 * self.subject_num * 0.8), 1, len(channels), 125))
                label = np.zeros(int(240 * self.subject_num * 0.8), dtype=int)
            else:
                # test data
                print("---------- 测试数据加载 ----------", flush=False)
                data = np.zeros((int(240 * self.subject_num * 0.2), 1, len(channels), 1375))
                pre_data = np.zeros((int(240 * self.subject_num * 0.2), 1, len(channels), 125))
                label = np.zeros(int(240 * self.subject_num * 0.2), dtype=int)

            start_subject = 1 if self.train else 29
            end_subject = 29 if self.train else 36
            for sub_num in tqdm(range(start_subject, end_subject)):
                f = scipy.io.loadmat(self.root + f"/S{sub_num}.mat")
                # print(f"mat{sub_num}文件大小: {f['data'].shape}")
                for block in range(6):
                    for target in range(40):
                        data[(sub_num - start_subject) * 240 + block * 40 + target] = f["data"][channels, 125:, target,
                                                                                      block]
                        pre_data[(sub_num - start_subject) * 240 + block * 40 + target] = f["data"][channels, :125,
                                                                                          target, block]
                        label[(sub_num - start_subject) * 240 + block * 40 + target] = int(target)
            return data, pre_data, label

        def __get_pre_filter(self, samp_rate):
            fs = samp_rate
            f0 = 50
            q = 35
            b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
            return b, a

        def __preprocess(self, data):
            filter_data = signal.filtfilt(self.filterB, self.filterA, data)
            return filter_data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, index) -> Tuple[Any, Any]:
            eeg, target = self.data[index], self.label[index]

            # 滤波处理
            eeg = self.__preprocess(eeg)

            if self.transform is not None:
                eeg = self.transform(eeg.copy())

            if self.target_transform is not None:
                target = self.target_transform(target.copy())

            eeg = eeg.float()
            return eeg, target

