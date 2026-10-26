from typing import Optional

import torch

from .kernels import center_gram, KernelFactory


def hsic(
    Z: torch.Tensor,
    Y: torch.Tensor,
    kernel_z="rbf",
    kernel_y="rbf",
    T: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    use_rff: bool = False,
    n_features: Optional[int] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.double,
    **kwargs,
) -> torch.Tensor:

    if kernel_z not in ["rbf", "rq"] and use_rff:
        raise ValueError("RFF only supported for stationary kernels (rbf and rq)")
    if kernel_y not in ["rbf", "rq"] and use_rff:
        raise ValueError("RFF only supported for stationary kernels (rbf and rq)")

    kernel_embedding = KernelFactory(kernel_z)
    kernel_target = KernelFactory(kernel_y)

    if use_rff:
        Kz, _ = kernel_embedding.forward_with_rff(X=Z, m=n_features, **kwargs)
        Ky, _ = kernel_target.forward_with_rff(X=Y, m=n_features, **kwargs)
    else:
        Kz, _ = kernel_embedding.forward(X=Z, **kwargs)
        Ky, _ = kernel_target.forward(X=Y, **kwargs)

    Kz = Kz.to(device=device, dtype=dtype)
    Ky = Ky.to(device=device, dtype=dtype)

    if T is not None:
        # Expect T: (n, m). Z should be (m, k) in this case.
        if T.dim() != 2:
            raise ValueError(f"T must be 2D (n,m). Got shape {tuple(T.shape)}")
        _, m = T.shape
        if Z.size(0) != m:
            raise ValueError(
                f"When T is provided, Z must have m rows. Got Z.shape[0]={Z.size(0)} but m={m}"
            )

        # Column-normalize T so each prototype defines a mixture over samples
        col_mass = T.sum(dim=0, keepdim=True).clamp_min(eps)  # (1, m)
        T_hat = T / col_mass  # (n, m)

        # Push label kernel to prototype space: (m, m)
        Ly = T_hat.t() @ Ky @ T_hat
    else:
        # Standard: (n, n) kernel on Y matches Z's n
        Ly = Ky
        if Z.size(0) != Y.size(0):
            raise ValueError(
                f"When T is None, Z and Y must have same number of rows. "
                f"Got Z.shape[0]={Z.size(0)} and Y.shape[0]={Y.size(0)}"
            )

    Kc = center_gram(Kz)
    Ly = Ly.to(Kc.dtype)  # ensure same dtype for centering and HSIC computation
    Lc = center_gram(Ly)

    # Normalized HSIC / CKA
    num = (Kc * Lc).sum()
    den = torch.linalg.norm(Kc, ord="fro") * torch.linalg.norm(Lc, ord="fro")
    return num / den.clamp_min(eps)

    # Unnormalized HSIC (biased estimator)
    # n = Z.size(0)
    # return (Kc * Lc).sum() / ((n - 1) ** 2)
