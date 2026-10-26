import torch


class RFFeatures(torch.nn.Module):
    """
    Random Fourier Features for an RBF kernel.
    φ: R^r -> R^p,   p = 2m
      φ(z) = sqrt(2/m) [cos(Ωz + b), sin(Ωz + b)]
    Jacobian available in closed form.

    This is a convenient concrete φ that matches the algorithm's need for J_φ(z).
    """

    def __init__(self, r: int, m: int, sigma: float = 1.0, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.r = r
        self.m = m
        self.p = 2 * m
        self.sigma = sigma

        # Ω ~ N(0, 1/sigma^2)
        Omega = torch.randn(m, r, **factory_kwargs) / sigma
        b = 2 * torch.pi * torch.rand(m, **factory_kwargs)
        self.register_buffer("Omega", Omega)  # (m, r)
        self.register_buffer("b", b)  # (m,)

        self.scale = (2.0 / m) ** 0.5

    def phi(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, r)
        returns Φ: (B, p)
        """
        T = X @ self.Omega.T + self.b  # (B, m)
        return self.scale * torch.cat([torch.cos(T), torch.sin(T)], dim=1)  # (B, 2m)
