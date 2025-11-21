# uncertainty_quantification/scripts/model_base.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(activation.lower(), nn.ReLU)
        layers = []
        d_prev = in_dim
        for d in hidden_dims:
            layers.append(nn.Linear(d_prev, d))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(d))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_prev = d
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if len(self.net) == 0:
            return x
        return self.net(x)

class RegressionHead(nn.Module):
    """
    head_type:
      - 'point': y_hat
      - 'gauss': (mu, sigma) with sigma = softplus(s) + eps
      - 'laplace': (mu, b) with b = softplus(s) + eps
    """
    def __init__(self, d_in: int, head_type: str = "point", eps: float = 1e-6):
        super().__init__()
        self.head_type = head_type.lower()
        self.eps = eps
        if self.head_type == "point":
            self.out = nn.Linear(d_in, 1)
        else:
            # two outputs: mean and unconstrained scale
            self.out = nn.Linear(d_in, 2)

    def forward(self, h):
        o = self.out(h)
        if self.head_type == "point":
            return {"mu": o[:, :1]}
        # split mean and raw scale
        mu = o[:, :1]
        s = o[:, 1:2]
        if self.head_type == "gauss":
            sigma = F.softplus(s) + self.eps
            return {"mu": mu, "sigma": sigma, "log_sigma": torch.log(sigma)}
        if self.head_type == "laplace":
            b = F.softplus(s) + self.eps
            return {"mu": mu, "b": b, "log_b": torch.log(b)}
        raise ValueError(f"Unsupported head_type={self.head_type}")

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], head_type: str = "point",
                 activation: str = "relu", dropout: float = 0.1, use_batchnorm: bool = True):
        super().__init__()
        self.backbone = MLP(in_dim, hidden_dims, activation, dropout, use_batchnorm)
        feat_dim = hidden_dims[-1] if len(hidden_dims) else in_dim
        self.head = RegressionHead(feat_dim, head_type=head_type)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)

# ---------- losses and helpers ----------
def mse_loss(mu, y):
    return F.mse_loss(mu, y)

def mae_loss(mu, y):
    return F.l1_loss(mu, y)

def huber_loss(mu, y, delta=1.0):
    return F.smooth_l1_loss(mu, y, beta=delta)

def gaussian_nll(mu, sigma, y):
    # 0.5 * [ ((y - mu)/sigma)^2 + 2 log sigma + log(2π) ]
    z = (y - mu) / torch.clamp(sigma, min=1e-8)
    nll = 0.5 * (z * z + 2.0 * torch.log(torch.clamp(sigma, min=1e-8)) + torch.log(torch.tensor(2.0 * 3.141592653589793)))
    return nll.mean()

def laplace_nll(mu, b, y):
    # |y - mu|/b + log(2b)
    b = torch.clamp(b, min=1e-8)
    nll = torch.abs(y - mu) / b + torch.log(2.0 * b)
    return nll.mean()
