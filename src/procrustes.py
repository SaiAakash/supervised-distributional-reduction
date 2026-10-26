import torch


@torch.no_grad()
def procrustes_fit(Z_ref, Z_hat, with_scale=True, eps=1e-12):
    """
    Fit Procrustes transform that maps Z_hat -> Z_ref.
    Returns a dict with (mu_ref, mu_hat, R, s).
    """
    Z_ref = Z_ref.detach()
    Z_hat = Z_hat.detach()

    # centers
    mu_ref = Z_ref.mean(dim=0, keepdim=True)
    mu_hat = Z_hat.mean(dim=0, keepdim=True)

    X = Z_hat - mu_hat  # source
    Y = Z_ref - mu_ref  # target

    # Cross-covariance
    C = X.T @ Y  # (k,k)

    # SVD: C = U S V^T  => R = U V^T  (maps X -> Y)
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    R = U @ Vh

    # Optional: enforce det(R)=+1 (avoid reflection) if you want a proper rotation
    if torch.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh

    if with_scale:
        # best scalar scale: s = tr((XR)^T Y) / ||X||^2
        XR = X @ R
        s = (XR * Y).sum() / (X.pow(2).sum().clamp_min(eps))
    else:
        s = torch.tensor(1.0, device=Z_ref.device, dtype=Z_ref.dtype)

    return {"mu_ref": mu_ref, "mu_hat": mu_hat, "R": R, "s": s}


@torch.no_grad()
def procrustes_apply(Z_hat, params):
    """
    Apply fitted transform to any Z_hat (train or test).
    """
    mu_ref, mu_hat, R, s = params["mu_ref"], params["mu_hat"], params["R"], params["s"]
    return (Z_hat - mu_hat) @ R * s + mu_ref
