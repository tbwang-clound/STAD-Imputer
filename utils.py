"""
Utility functions for STAD-Imputer.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def check_dir(path: str) -> bool:
    """Create directory if it does not exist. Returns True if it already existed."""
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True


# ---------------------------------------------------------------------------
# Masked metrics
# ---------------------------------------------------------------------------

def masked_mae(preds: torch.Tensor, labels: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error over masked positions."""
    residual = torch.abs(labels - preds) * mask
    num_eval = mask.sum()
    return residual.sum() / (num_eval if num_eval > 0 else 1)


def masked_mse(preds: torch.Tensor, labels: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error over masked positions."""
    residual = (labels - preds) * mask
    num_eval = mask.sum()
    return (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)


def masked_mape(preds: torch.Tensor, labels: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Percentage Error over masked positions."""
    mask = mask.float()
    mask = mask / (torch.mean(mask) + 1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / (torch.abs(labels) + 1e-5)
    loss = loss * mask
    return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss))


def masked_r2(preds: torch.Tensor, labels: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Coefficient of determination R² over masked positions."""
    mask = mask.bool()
    p = preds[mask]
    l = labels[mask]
    if l.numel() == 0:
        return torch.tensor(0.0)
    res = torch.sum((l - p) ** 2)
    tot = torch.sum((l - torch.mean(l)) ** 2)
    return 1 - res / (tot + 1e-5)


def masked_ssim(preds: torch.Tensor, labels: torch.Tensor,
                mask: torch.Tensor,
                c1: float = 1e-4, c2: float = 9e-4) -> torch.Tensor:
    """Structural Similarity Index over masked positions."""
    mask = mask.bool()
    p = preds[mask]
    l = labels[mask]
    if l.numel() < 2:
        return torch.tensor(0.0)
    mu_x, mu_y = torch.mean(l), torch.mean(p)
    sig_x, sig_y = torch.var(l), torch.var(p)
    sig_xy = torch.mean((l - mu_x) * (p - mu_y))
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2)
    return num / den


def calculate_crps(samples: torch.Tensor, labels: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Continuous Ranked Probability Score (CRPS) via quantile approximation.

    Args:
        samples: (B, num_samples, T, N)
        labels:  (B, T, N)
        mask:    (B, T, N)
    """
    mask = mask.bool()
    if not mask.any():
        return torch.tensor(0.0)

    target = labels[mask]                          # (M,)
    s = samples.transpose(0, 1)[:, mask]           # (num_samples, M)
    s = torch.sort(s, dim=0)[0]

    alphas = torch.linspace(0.05, 0.95, 19).to(samples.device)
    num_samples = s.size(0)
    crps_sum = 0.0
    for alpha in alphas:
        idx = int(alpha * num_samples)
        q = s[idx]
        indicator = (target < q).float()
        crps_sum += ((alpha - indicator) * (target - q)).mean()

    return 2 * crps_sum / 19.0


# ---------------------------------------------------------------------------
# Model size report
# ---------------------------------------------------------------------------

def get_model_size_info(model: nn.Module) -> dict:
    """Print and return model parameter statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)
    print("=" * 60)
    print(f"Model:            {model.__class__.__name__}")
    print(f"Total params:     {total / 1e6:.2f} M")
    print(f"Trainable params: {trainable / 1e6:.2f} M")
    print(f"Memory (FP32):    {size_mb:.2f} MB")
    print("=" * 60)
    return {"total_params": total, "trainable_params": trainable,
            "param_size_mb": size_mb}
