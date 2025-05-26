"""
This module implements a collection of deep learning models for feature processing:

1. CNN: A convolutional neural network for feature extraction.
2. Classifier series: Multiple classifiers designed for feature classification.
3. RandMix: A random mixing module for data augmentation.
4. AdaIN1d: A 1D Adaptive Instance Normalization layer for feature normalization.
5. Masker: A module for category-specific feature masking.
6. Other utility functions: Including contrastive learning losses (e.g., CLUB) and more.
"""


import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class CNN(nn.Module):
     """CNN models for feature extraction

    Args:
        n_classes (int): Number of classification categories

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
            x: input tensor
            tau: Temperature parameters

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
        """Forward propagation of models

        Args:
            x: input tensor
            tau: Temperature parameters

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
    """basic classifier model

    Args:
        n_classes: Number of classification categories
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
    """One-dimensional adaptive instance normalization layer

    Args:
        style_dim: Style Dimension
        num_features: Number of features
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
    """RandMix Module

    Args:
        noise_lv: noise level
    """

    def __init__(self, noise_lv: float):
        super().__init__()
        self.zdim = 3
        self.noise_lv = noise_lv

        # AdaIN layer
        self.adain_layers = nn.ModuleList([
            AdaIN1d(self.zdim, 1) for _ in range(4)
        ])

        # space transformation layer
        self.spatial_transforms = nn.ModuleList([
            nn.ModuleDict({
                'down': nn.Conv1d(1, 1, 2 * i + 3),
                'up': nn.ConvTranspose1d(1, 1, 2 * i + 3)
            }) for i in range(4)
        ])

        self.mixing_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Fixed blend weights
        # self.random_weights = torch.randn(5)            # Random weighting

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        Args:
            x: input tensor 

        Returns:
            mixed: Mixed features
        """
        original = x
        x = x + torch.randn_like(x) * self.noise_lv * 0.001

        spatial_features = []
        for i, (transform, adain) in enumerate(zip(self.spatial_transforms, self.adain_layers)):
            down = transform['down'](x)
            style = torch.randn(len(down), self.zdim, device=x.device)
            transformed = adain(down, style)
            spatial_features.append(torch.relu(transform['up'](transformed)))

        mixed_features = sum(w * f for w, f in zip(self.mixing_weights[:4], spatial_features))
        original_weighted = self.mixing_weights[4] * original
        mixed = mixed_features + original_weighted

        return mixed


def loglikeli(mu: torch.Tensor, logvar: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
    """Compute the log-likelihood

    Args:
        mu: average values
        logvar: logarithmic variance
        y_samples: samples

    Returns:
        log_likelihood
    """
    return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).mean()


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor, factor: float = 0.2) -> torch.Tensor:
    """reparameterization technique

    Args:
        mu: average values
        logvar: logarithmic variance
        factor: scaling factor

    Returns:
        Samples after reparameterization
    """
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + factor * std * eps


def club(mu: torch.Tensor, logvar: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
    """CLUB (Contrastive Log-ratio Upper Bound)

    Args:
        mu: average values
        logvar: logarithmic variance
        y_samples: samples

    Returns:
        upper_bound: CLUB
    """
    sample_size = y_samples.shape[0]
    random_index = torch.randperm(sample_size).long()

    positive = -(mu - y_samples) ** 2 / logvar.exp()
    negative = -(mu - y_samples[random_index]) ** 2 / logvar.exp()
    upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    return upper_bound / 2.
