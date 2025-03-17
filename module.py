"""
特征处理模型集合 (Feature Processing Models)

本模块实现了一系列用于特征处理的深度学习模型：
1. CNN: 用于特征提取的卷积神经网络
2. Classifier系列: 用于特征分类的多个分类器模型
3. RandMix: 用于数据增强的随机混合模块
4. AdaIN1d: 用于特征归一化的一维自适应实例归一化层
5. Masker: 用于类别特征掩码的模块
6. 其他工具函数: 包括对比学习损失(CLUB)等

"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class CNN(nn.Module):
    """卷积神经网络模型，用于特征提取

    Args:
        n_classes (int): 分类类别数

    """

    def __init__(self, n_classes: int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 32),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.feature_layer = nn.Sequential(
            nn.Conv1d(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        self.conv5_branches = nn.ModuleDict({
            'branch1': nn.Sequential(
                nn.Conv1d(128, 128, 3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ),
            'branch2': nn.Sequential(
                nn.Conv1d(128, 128, 5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ),
            'branch3': nn.Sequential(
                nn.Conv1d(128, 128, 9),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
        })

        # Channel mask
        self.channel_mask = nn.Parameter(torch.randn(2, 32, 1))

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 6, 128),
            nn.Linear(128, n_classes)
        )

    def forward_first_layer(self, x: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation layer 1, returning domain invariant and domain specific features

        Args:
            x: 输入张量
            tau: 温度参数

        Returns:
            f_invariant: domain invariant features
            f_specific: domain specific features
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        channel_weights = torch.softmax(self.channel_mask / tau, dim=0)
        f_invariant = x2 * channel_weights[0].view(1, *channel_weights[0].shape)
        f_specific = x2 * channel_weights[1].view(1, *channel_weights[1].shape)

        return f_invariant, f_specific

    def forward(self, x: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """模型的前向传播

        Args:
            x: 输入张量
            tau: 温度参数

        Returns:
            features
            channel_weights: Weighting of the channel mask
            f_invariant: domain invariant features
            f_specific: domain specific features
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        channel_weights = torch.softmax(self.channel_mask / tau, dim=0)
        content_features = x2 * channel_weights[0].view(1, *channel_weights[0].shape)
        style_features = x2 * channel_weights[1].view(1, *channel_weights[1].shape)

        x3 = self.feature_layer(content_features)
        x4 = self.conv4(x3)
        x5 = self.conv5_branches['branch1'](x4)

        features = x5.view(x5.size(0), -1)

        return features, channel_weights, content_features, style_features

class Classifier(nn.Module):
    """基础分类器模型

    Args:
        n_classes: 分类类别数
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(128 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)


class Classifier_ad(nn.Module):
    """adversarial classifier
    """
    def __init__(self, n_classes):
        super(Classifier_ad, self).__init__()

        self.fc = nn.Sequential(nn.Linear(128 * 6, 128 * 2))
        self.fc1 = nn.Sequential(nn.Linear(128 * 2, 128))
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):

        fea = self.fc(x)
        fea = self.fc1(fea)
        label = self.out(fea)
        return label


class Classifier_all(nn.Module):

    def __init__(self, n_classes):
        super(Classifier_all, self).__init__()

        self.fc = nn.Sequential(nn.Linear(128 * 6, 128 * 2))
        self.fc1 = nn.Sequential(nn.Linear(128 * 2, 128))
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):

        fea = self.fc(x)
        fea = self.fc1(fea)
        label = self.out(fea)
        return label


class Projector(nn.Module):
    def __init__(self, output_size=1024):
        super(Projector, self).__init__()
        self.conv = nn.Conv1d(32, 64, 3)
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv1 = nn.Conv1d(64, 128, 3)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.fc = nn.Linear(3904, output_size)
        self.p_logvar = nn.Sequential(nn.Linear(3904, 1024))
        self.p_mu = nn.Sequential(nn.Linear(3904, 1024))
        self.fc1 = nn.Linear(output_size, 128)

    def forward(self, x, train=True):

        end_points = {}
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        logvar = self.p_logvar(x)
        mu = self.p_mu(x)

        end_points['logvar'] = logvar

        end_points['mu'] = mu

        if train:
            x = reparametrize(mu, logvar)
        else:
            x = mu

        end_points['Embedding'] = x

        return x, end_points


class Masker(nn.Module):
    def __init__(self, in_dim=128 * 6, num_classes=128 * 6, middle=256 * 6, k=256):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k
        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
        mask = self.bn(self.layers(f))
        z = torch.zeros_like(mask)
        for _ in range(self.k):
            mask = F.gumbel_softmax(mask, dim=1, tau=0.1, hard=False)
            z = torch.maximum(mask, z)
        return z


class AdaIN1d(nn.Module):
    """一维自适应实例归一化层

    Args:
        style_dim: 风格维度
        num_features: 特征数量
    """

    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.style_transform = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        style_params = self.style_transform(style)
        style_params = style_params.view(style_params.size(0), style_params.size(1), 1)
        gamma, beta = torch.chunk(style_params, chunks=2, dim=1)
        normalized = self.norm(x)
        return (1 + gamma) * normalized + beta


class RandMix(nn.Module):
    """随机混合增强模块

    Args:
        noise_lv: 噪声水平
    """

    def __init__(self, noise_lv: float):
        super().__init__()
        self.zdim = 3
        self.noise_lv = noise_lv

        # AdaIN层
        self.adain_layers = nn.ModuleList([
            AdaIN1d(self.zdim, 1) for _ in range(4)
        ])

        # 空间变换层
        self.spatial_transforms = nn.ModuleList([
            nn.ModuleDict({
                'down': nn.Conv1d(1, 1, 2 * i + 3),
                'up': nn.ConvTranspose1d(1, 1, 2 * i + 3)
            }) for i in range(4)
        ])

        self.mixing_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Fixed blend weights
        # self.random_weights = torch.randn(5)            # Random weighting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量

        Returns:
            mixed: 混合后的特征
        """
        original = x
        x = x + torch.randn_like(x) * self.noise_lv * 0.001

        # 空间变换
        spatial_features = []
        for i, (transform, adain) in enumerate(zip(self.spatial_transforms, self.adain_layers)):
            down = transform['down'](x)
            style = torch.randn(len(down), self.zdim, device=x.device)
            transformed = adain(down, style)
            spatial_features.append(torch.relu(transform['up'](transformed)))

        # 特征混合
        mixed_features = sum(w * f for w, f in zip(self.mixing_weights[:4], spatial_features))
        original_weighted = self.mixing_weights[4] * original
        mixed = mixed_features + original_weighted

        return mixed


def loglikeli(mu: torch.Tensor, logvar: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
    """计算对数似然

    Args:
        mu: 均值
        logvar: 对数方差
        y_samples: 样本

    Returns:
        log_likelihood: 对数似然值
    """
    return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).mean()


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor, factor: float = 0.2) -> torch.Tensor:
    """重参数化技巧

    Args:
        mu: 均值
        logvar: 对数方差
        factor: 缩放因子

    Returns:
        重参数化后的样本
    """
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + factor * std * eps


def club(mu: torch.Tensor, logvar: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
    """计算CLUB (Contrastive Log-ratio Upper Bound)

    Args:
        mu: 均值
        logvar: 对数方差
        y_samples: 样本

    Returns:
        upper_bound: CLUB上界
    """
    sample_size = y_samples.shape[0]
    random_index = torch.randperm(sample_size).long()

    positive = -(mu - y_samples) ** 2 / logvar.exp()
    negative = -(mu - y_samples[random_index]) ** 2 / logvar.exp()
    upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    return upper_bound / 2.