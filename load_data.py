# -*- coding: utf-8 -*-
import torch
import scipy.io as scio
import numpy as np
from scipy.fftpack import fft

def load_data(path, temp, com_class, class_sample, mis_class):
    # 1. 加载 .mat 数据文件并获取指定键的数据
    mat_data = scio.loadmat(path)
    data = mat_data.get(temp)
    if data is None:
        raise KeyError(f"Key '{temp}' not found in the MATLAB file.")

    n_class = com_class - len(mis_class)  # 有效类别数量（总类数减去排除的类）

    # 2. 删除需要排除的类别数据行
    if len(mis_class) > 0:
        # 提取每行数据的标签列（最后一列）
        labels = data[:, 0, -1] if data.ndim == 3 else data[..., -1]
        # 构建掩码，标记需保留的样本（标签不在排除列表中）
        mask = ~np.isin(labels, mis_class)
        data = data[mask]  # 仅保留未被排除的样本

    # 3. 划分训练集和测试集数据
    train_count = class_sample * n_class
    train_data = data[:train_count]
    test_data = data[train_count:]
    # 分离特征和标签
    train_x = train_data[..., :-1]  # 所有特征列
    train_y = train_data[..., -1]  # 标签列
    test_x = test_data[..., :-1]
    test_y = test_data[..., -1]
    # 如果标签仍含有多维（例如数据原本有额外维度），将其压缩为一维
    if train_y.ndim > 1:
        train_y = train_y[:, 0]
    if test_y.ndim > 1:
        test_y = test_y[:, 0]

    # 4. 定义数据预处理函数：FFT 变换并截取一半频谱
    def transform_signal(x):
        # 对信号最后一维进行 FFT
        X_fft = 2 * np.abs(fft(x).real) / x.shape[-1]
        # 仅保留前半部分频谱（去除对称的后半部分）
        X_fft = X_fft[..., :x.shape[-1] // 2]
        return X_fft

    # 5. 对训练集和测试集特征进行 FFT 变换
    train_x = transform_signal(train_x)
    test_x = transform_signal(test_x)

    # 6. 特征归一化：使用训练集的最小值和最大值将特征缩放到 [0, 1]
    min_val = train_x.min()
    max_val = train_x.max()
    train_x = (train_x - min_val) / (max_val - min_val)
    test_x = (test_x - min_val) / (max_val - min_val)

    # 7. 转换为 PyTorch 的 FloatTensor 类型
    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(test_y)

    return train_x, train_y, test_x, test_y



