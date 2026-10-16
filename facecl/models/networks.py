"""Backbones used in the CL literature we compare against.

* ResNet-18 as in DER / Mammoth for 32x32 and 64x64 inputs: 3x3 stem, no max-pool,
  width nf=64 (about 11M parameters), global average pooling.
* MLP with two hidden layers of 100 units for MNIST variants (as in DER / GEM).

Every model exposes `forward(x, return_features=False)` and `.feat_dim`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, n_classes, in_channels=3, nf=64, mean=(0.0,), std=(1.0,), dropout=0.0):
        super().__init__()
        self.norm = Normalize(mean, std)
        self.conv1 = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        layers, in_planes = [], nf
        for i, (planes, stride) in enumerate([(nf, 1), (2 * nf, 2), (4 * nf, 2), (8 * nf, 2)]):
            layers.append(nn.Sequential(BasicBlock(in_planes, planes, stride), BasicBlock(planes, planes, 1)))
            in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.feat_dim = 8 * nf
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(self.norm(x))))
        out = self.layers(out)
        return torch.flatten(F.adaptive_avg_pool2d(out, 1), 1)

    def forward(self, x, return_features=False):
        feats = self.features(x)
        logits = self.classifier(self.dropout(feats))
        return (logits, feats) if return_features else logits


class MLP(nn.Module):
    def __init__(self, n_classes, in_dim=784, hidden=100, mean=(0.0,), std=(1.0,), dropout=0.0):
        super().__init__()
        self.norm = Normalize(mean, std)
        self.body = nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.feat_dim = hidden
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden, n_classes)

    def features(self, x):
        return self.body(self.norm(x))

    def forward(self, x, return_features=False):
        feats = self.features(x)
        logits = self.classifier(self.dropout(feats))
        return (logits, feats) if return_features else logits


def build_model(cfg: dict, bench) -> nn.Module:
    arch = cfg.get("backbone", "resnet18")
    c, h, w = bench.in_shape
    dropout = float(cfg.get("dropout", 0.0))
    if arch == "resnet18":
        return ResNet18(bench.n_classes, c, int(cfg.get("nf", 64)), bench.mean, bench.std, dropout)
    if arch == "mlp":
        return MLP(bench.n_classes, c * h * w, int(cfg.get("hidden", 100)), bench.mean, bench.std, dropout)
    raise ValueError(f"unknown backbone {arch}")
