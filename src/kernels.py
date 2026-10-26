from typing import Optional, Union
import warnings

import torch

from src.rff import RFFeatures


class KernelFactory:
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def forward(
        self, X: torch.Tensor, Y: Optional[torch.Tensor] = None, **kwargs
    ) -> Union[torch.Tensor, Optional[torch.Tensor]]:
        # Compute Gram Matrices when Y is None
        if Y is None:
            Y = X

        # Unpack kernel-specific parameters from kwargs
        if kwargs is not None:
            sigma = kwargs.get("sigma", None)
            eps = kwargs.get("eps", 1e-8)
            alpha = kwargs.get("alpha", 1.0)
        else:
            kwargs = {"sigma": None, "eps": 1e-8, "alpha": 1.0}

        if self.kernel_type == "linear":
            return linear_kernel(X, Y)
        elif self.kernel_type == "rbf":
            return rbf_kernel(X, Y, sigma=sigma, eps=eps)
        elif self.kernel_type == "rq":
            return rq_kernel(X, Y, sigma=sigma, alpha=alpha, eps=eps)
        elif self.kernel_type == "delta":
            return delta_kernel(X)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def forward_with_rff(
        self, X: torch.Tensor, m: int, Y: Optional[torch.Tensor] = None, **kwargs
    ) -> Union[torch.Tensor, Optional[torch.Tensor]]:
        # Compute Gram Matrices when Y is None
        if Y is None:
            Y = X

        if self.kernel_type == "rbf":
            return rbf_with_rff_kernel(X, m, Y, **kwargs)
        elif self.kernel_type == "rq":
            raise NotImplementedError(
                "RFF for the Rational Quadratic Kernel will be implemented in the next release"
            )
        else:
            raise ValueError("Only RBF is supported with a Random Fourier Feature Approximation")


def center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    H = torch.eye(n, device=K.device, dtype=K.dtype) - (1.0 / n) * torch.ones(
        (n, n), device=K.device, dtype=K.dtype
    )
    return H @ K @ H


def linear_kernel(
    X: torch.Tensor, Y: Optional[torch.Tensor] = None
) -> Union[torch.Tensor, Optional[torch.Tensor]]:
    return X @ Y.t(), None


def rbf_kernel(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    sigma: Optional[float] = None,
    eps: float = 1e-8,
) -> Union[torch.Tensor, Optional[torch.Tensor]]:
    """
    X: (n, d), Y: (m, d) -> K: (n, m)
    """
    X2 = (X * X).sum(dim=1, keepdim=True)  # (n, 1)
    Y2 = (Y * Y).sum(dim=1, keepdim=True).T  # (1, m)
    dist2 = X2 + Y2 - 2.0 * (X @ Y.T)
    dist2 = dist2.clamp_min(0.0)  # numerical stability

    if sigma is None:
        with torch.no_grad():
            vals = dist2[torch.triu(torch.ones_like(dist2), diagonal=1) > 0]
            med = torch.median(vals)
            sigma2 = med.clamp_min(eps)
    else:
        sigma2 = torch.tensor(sigma * sigma).clamp_min(eps)
    return torch.exp(-dist2 / (2.0 * sigma2)), torch.sqrt(sigma2)


def rbf_with_rff_kernel(
    X: torch.Tensor,
    m: int,
    Y: Optional[torch.Tensor] = None,
    sigma: Optional[float] = 1.0,
) -> Union[torch.Tensor, Optional[torch.Tensor]]:
    """
    m: int
        Number of Fourier Features (Fourier basis functions)
    sigma: float
        Standard deviation of the Normal distribution from which the frequencies omega are sampled
    """
    r = X.shape[-1]
    rff = RFFeatures(r=r, m=m, sigma=sigma)
    phi_X = rff.phi(X)
    phi_y = rff.phi(Y)
    return phi_X.T @ phi_y, rff


def rq_kernel(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    sigma: Optional[float] = None,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> Union[torch.Tensor, Optional[torch.Tensor]]:
    """
    X: (n, d)
    Y: (m, d) or None (if None, Y = X)
    lengthscale: ℓ > 0
    alpha: α > 0

    Returns:
        K: (n, m) kernel matrix
    """
    # Squared Euclidean distances
    X_norm = (X**2).sum(dim=1, keepdim=True)  # (n, 1)
    Y_norm = (Y**2).sum(dim=1, keepdim=True).T  # (1, m)
    dist2 = X_norm + Y_norm - 2 * X @ Y.T  # (n, m)
    dist2 = dist2.clamp_min(0.0)  # numerical stability

    if sigma is None:
        with torch.no_grad():
            vals = dist2[torch.triu(torch.ones_like(dist2), diagonal=1) > 0]
            med = torch.median(vals)
            sigma2 = med.clamp_min(eps)
    else:
        sigma2 = torch.tensor(sigma * sigma).clamp_min(eps)

    K = (1 + dist2 / (2 * alpha * sigma2)) ** (-alpha)
    return K, torch.sqrt(sigma2)


def delta_kernel(Y: torch.Tensor) -> Union[torch.Tensor, Optional[torch.Tensor]]:
    """
    y: (n,) tensor of categorical labels (int or long)
    returns: (n, n) delta kernel matrix
    """
    # Case where Y is already one-hot encoded: (n, c) with c > 1. In this case we can just do a linear kernel on the one-hot vectors.
    if Y.ndim == 2 and Y.shape[-1] > 1:
        K = (Y @ Y.T)  # (n, n)
        return K, None
    elif Y.ndim == 1:
        Y = Y.view(-1, 1)  # (n, 1)
        K = (Y == Y.T)  # (n, n)
        return K, None
