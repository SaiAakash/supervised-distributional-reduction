import numpy as np
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.kernels import KernelFactory, delta_kernel


class KernelSupervisedPCA(BaseEstimator, TransformerMixin):
    """
    Kernel Supervised PCA (KSPCA).

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If None, keeps all.
    kernel : str, optional
        Input kernel for X. Supported: "rbf", "linear", "precomputed".
    gamma : float, optional
        Input RBF coefficient in exp(-gamma * ||x - x'||^2).
    degree : int, optional
        Reserved for compatibility. Not used.
    coef0 : float, optional
        Reserved for compatibility. Not used.
    target_kernel : {"auto", "delta", "rbf", "precomputed"}, optional
        Kernel for labels/targets.
    target_gamma : float, optional
        Target RBF coefficient in exp(-target_gamma * ||y - y'||^2).
    reg : float, optional
        Tikhonov regularization added to K in the generalized eigenproblem.
    device : str or torch.device, optional
        Compute device.
    rff_n_features : int, optional
        Reserved for backward compatibility. Unused because RBF kernels are
        computed with the vanilla kernel implementation.
    target_rff_n_features : int, optional
        Reserved for backward compatibility. Unused because target RBF kernels
        are computed with the vanilla kernel implementation.
    """

    def __init__(
        self,
        n_components=None,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1.0,
        target_kernel="auto",
        target_gamma=None,
        reg=1e-6,
        device="cpu",
        rff_n_features=256,
        target_rff_n_features=None,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.target_kernel = target_kernel
        self.target_gamma = target_gamma
        self.reg = reg
        self.device = device
        self.rff_n_features = rff_n_features
        self.target_rff_n_features = target_rff_n_features

    def fit(self, X, y):
        if y is None:
            raise ValueError("y must be provided for supervised PCA.")

        K, X_fit = self._compute_input_kernel(X, fit=True)
        n = K.shape[0]

        if self.n_components is None:
            n_components = n
        else:
            n_components = int(self.n_components)
            if n_components <= 0 or n_components > n:
                raise ValueError(
                    "n_components must be in [1, n_samples]. "
                    f"Got n_components={n_components}, n_samples={n}."
                )

        L = self._compute_target_kernel(y, n_samples=n)
        Lc = self._center_kernel(L)

        Q = K @ Lc @ K
        Q = 0.5 * (Q + Q.T)
        K_reg = K + self.reg * torch.eye(n, device=K.device, dtype=K.dtype)
        K_reg = 0.5 * (K_reg + K_reg.T)

        eigvals, eigvecs = self._generalized_eigh(Q, K_reg)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.components_ = eigvecs[:, :n_components]
        self.eigenvalues_ = eigvals[:n_components]
        self.n_components_ = n_components
        self.n_samples_ = n
        self.K_fit_ = K
        self.X_fit_ = X_fit
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        Ktest, _ = self._compute_input_kernel(X, fit=False)
        Z = (self.components_.T @ Ktest).T
        if self._return_numpy_ and not isinstance(X, torch.Tensor):
            return Z.cpu().numpy()
        return Z

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X if self.kernel != "precomputed" else self.K_fit_)

    def _compute_input_kernel(self, X, fit):
        device = self._resolve_device(X)
        dtype = X.dtype if isinstance(X, torch.Tensor) else torch.float64
        self._return_numpy_ = device.type == "cpu" and not isinstance(X, torch.Tensor)

        if self.kernel == "precomputed":
            X_arr = np.asarray(X) if not isinstance(X, torch.Tensor) else X
            if fit:
                if X_arr.ndim != 2 or X_arr.shape[0] != X_arr.shape[1]:
                    raise ValueError("With kernel='precomputed', X must be a square kernel matrix.")
            else:
                if X_arr.ndim != 2 or X_arr.shape[0] != self.n_samples_:
                    raise ValueError(
                        "With kernel='precomputed', X must be a kernel matrix of shape "
                        f"(n_train, n_test) = ({self.n_samples_}, n_test)."
                    )
            K = self._to_tensor(X_arr, device=device, dtype=dtype)
            return K, None

        X_tensor = self._to_tensor(X, device=device, dtype=dtype)
        if self.kernel not in {"linear", "rbf"}:
            raise ValueError("kernel must be one of {'linear', 'rbf', 'precomputed'}.")

        if fit:
            kernel = KernelFactory(self.kernel)
            K, _ = kernel.forward(
                X=X_tensor,
                sigma=self._gamma_to_sigma(self.gamma, X_tensor.shape[1])
                if self.kernel == "rbf"
                else None,
            )
            return K, X_tensor

        if self.X_fit_ is None:
            raise ValueError("X_fit_ is missing; cannot compute kernel for new data.")

        kernel = KernelFactory(self.kernel)
        Ktest, _ = kernel.forward(
            X=self.X_fit_,
            Y=X_tensor,
            sigma=self._gamma_to_sigma(self.gamma, self.X_fit_.shape[1])
            if self.kernel == "rbf"
            else None,
        )
        return Ktest, None

    def _compute_target_kernel(self, y, n_samples):
        device = self._resolve_device(y)
        dtype = y.dtype if isinstance(y, torch.Tensor) else torch.float64

        if self.target_kernel == "precomputed":
            L = np.asarray(y) if not isinstance(y, torch.Tensor) else y
            if L.ndim != 2 or L.shape[0] != L.shape[1]:
                raise ValueError(
                    "With target_kernel='precomputed', y must be a square kernel matrix."
                )
            if L.shape[0] != n_samples:
                raise ValueError(
                    "Precomputed target kernel has wrong shape. "
                    f"Expected ({n_samples}, {n_samples}), got {L.shape}."
                )
            return self._to_tensor(L, device=device, dtype=dtype)

        y_arr = None
        if isinstance(y, torch.Tensor):
            y_tensor = y.to(device=device, dtype=dtype)
        else:
            y_arr = np.asarray(y)
            if np.issubdtype(y_arr.dtype, np.number):
                y_tensor = torch.as_tensor(y_arr, device=device, dtype=dtype)
            else:
                y_tensor = None

        if y_tensor is not None:
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.view(-1, 1)
            if y_tensor.shape[0] != n_samples:
                raise ValueError(
                    f"y has {y_tensor.shape[0]} samples, but X has {n_samples} samples."
                )
        else:
            if y_arr is None:
                y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            if y_arr.shape[0] != n_samples:
                raise ValueError(f"y has {y_arr.shape[0]} samples, but X has {n_samples} samples.")

        target_kernel = self.target_kernel
        if target_kernel == "auto":
            if y_tensor is not None:
                if y_tensor.dtype in {
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.bool,
                }:
                    target_kernel = "delta"
                else:
                    target_kernel = "rbf"
            elif np.issubdtype(y_arr.dtype, np.integer) or y_arr.dtype.kind in {
                "U",
                "S",
                "O",
            }:
                target_kernel = "delta"
            else:
                target_kernel = "rbf"

        if target_kernel == "delta":
            labels = self._delta_labels_tensor(y_tensor, y_arr, device=device, dtype=dtype)
            L, _ = delta_kernel(labels)
            return L.to(dtype=dtype, device=device)

        if target_kernel == "rbf":
            if y_tensor is None:
                raise ValueError("RBF target kernel requires numeric targets.")
            kernel = KernelFactory("rbf")
            L, _ = kernel.forward(
                X=y_tensor,
                sigma=self._gamma_to_sigma(self.target_gamma, y_tensor.shape[1]),
            )
            return L

        raise ValueError("target_kernel must be one of {'auto', 'delta', 'rbf', 'precomputed'}.")

    @staticmethod
    def _delta_labels_tensor(y_tensor, y_arr, device, dtype):
        if y_tensor is not None:
            if y_tensor.ndim == 1 or y_tensor.shape[1] == 1:
                return y_tensor.reshape(-1)
            _, inverse = torch.unique(y_tensor, dim=0, return_inverse=True)
            return inverse.to(device=device, dtype=dtype)

        if y_arr.ndim == 1 or y_arr.shape[1] == 1:
            encoded = np.unique(y_arr.reshape(-1), return_inverse=True)[1]
            return torch.as_tensor(encoded, device=device, dtype=dtype)

        rows = np.ascontiguousarray(y_arr).view(
            np.dtype((np.void, y_arr.dtype.itemsize * y_arr.shape[1]))
        )
        encoded = np.unique(rows, return_inverse=True)[1]
        return torch.as_tensor(encoded, device=device, dtype=dtype)

    @staticmethod
    def _center_kernel(K):
        row_mean = K.mean(dim=1, keepdim=True)
        col_mean = K.mean(dim=0, keepdim=True)
        total_mean = K.mean()
        return K - row_mean - col_mean + total_mean

    @staticmethod
    def _gamma_to_sigma(gamma, input_dim):
        if gamma is None:
            gamma = 1.0 / input_dim
        if gamma <= 0:
            raise ValueError(f"gamma must be positive for RBF kernels, got {gamma}.")
        return float((1.0 / (2.0 * gamma)) ** 0.5)

    @staticmethod
    def _generalized_eigh(A, B):
        # Solve Av=lambda Bv via symmetric reduction:
        # C = L^{-1} A L^{-T}, C u = lambda u, v = L^{-T} u
        try:
            L = torch.linalg.cholesky(B)
            LinvA = torch.linalg.solve_triangular(L, A, upper=False, left=True)
            C = torch.linalg.solve_triangular(L, LinvA.T, upper=False, left=True).T
            C = 0.5 * (C + C.T)
            eigvals, U = torch.linalg.eigh(C)
            eigvecs = torch.linalg.solve_triangular(L.T, U, upper=True, left=True)
            return eigvals, eigvecs
        except RuntimeError:
            # Fallback for non-PD precomputed kernels.
            eigvals, eigvecs = torch.linalg.eig(torch.linalg.solve(B, A))
            return eigvals.real, eigvecs.real

    def _resolve_device(self, X):
        if self.device is None:
            if isinstance(X, torch.Tensor):
                return X.device
            return torch.device("cpu")
        return torch.device(self.device)

    @staticmethod
    def _to_tensor(X, device, dtype):
        if isinstance(X, torch.Tensor):
            return X.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(X), device=device, dtype=dtype)
