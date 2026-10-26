# Affinity matrices
import torch
import math
from tqdm import tqdm
from abc import abstractmethod
from src.utils import entropy, false_position
from src.utils_hyperbolic import minkowski_ip2
import numpy as np


OPTIMIZERS = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "NAdam": torch.optim.NAdam,
}


class NanError(Exception):
    pass


class BadPerplexity(Exception):
    pass


class BaseAffinity:
    def __init__(self, tolog=False):
        self.tolog = tolog
        self.log_ = {}

    @abstractmethod
    def compute_affinity(self, X):
        pass

    def compute_log_affinity(self, X):
        return torch.log(self.compute_affinity(X) + 1e-10)


class MinkwskiInnerProductAffinity(BaseAffinity):
    def compute_affinity(self, X):
        self.data = -minkowski_ip2(X, X)
        return self.data


class GramAffinity(BaseAffinity):
    def __init__(self, centering=False, tolog=False, normalize="trace", eps=1e-12):
        super().__init__(tolog=tolog)
        self.centering = centering
        self.normalize = normalize
        self.eps = eps

    def _normalize(self, K):
        if self.normalize is None:
            return K
        if self.normalize == "trace":
            return K / (torch.trace(K).clamp_min(self.eps))
        if self.normalize == "fro":
            return K / (K.norm(p="fro").clamp_min(self.eps))
        raise ValueError(f"Unknown normalize={self.normalize}")

    def compute_affinity(self, X):
        if len(X.shape) == 3:  # low rank case
            if self.centering:
                X = X - X.mean(1, keepdim=True)
            K = X[0] @ X[1].T
            return self._normalize(K)
        else:
            if self.centering:
                X = X - X.mean(0, keepdim=True)
            K = X @ X.T
            return self._normalize(K)


class LogAffinity(BaseAffinity):
    def compute_log_affinity(self, X):
        raise NotImplementedError("Subclasses must implement compute_log_affinity")

    def compute_affinity(self, X):
        return torch.exp(self.compute_log_affinity(X))


class DiffusionAffinity(BaseAffinity):
    def compute_log_affinity(self, X):
        raise NotImplementedError("Subclasses must implement compute_log_affinity")

    def compute_affinity(self, X):
        """
        Computes an affinity matrix from an affinity matrix in log space
        Normalize it to make a probablistic transition matrix, aka diffusion operator

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix.
        """
        log_P = self.compute_log_affinity(X)
        self.data = torch.exp(log_P)
        # normalize it to a transition kernel
        marg1 = self.data.sum(1)
        self.data = marg1[None, :] * (self.data / marg1[:, None])
        marg2 = torch.sqrt(self.data.sum(1))
        self.data = marg2[None, :] * (self.data / marg2[:, None])
        return self.data


class LorentzHyperbolicAffinity(LogAffinity):
    def __init__(self, beta=1.0, p=2):
        self.beta = beta
        self.p = p
        super(LorentzHyperbolicAffinity, self).__init__()

    def compute_log_affinity(self, X):
        return (
            -self.beta * torch.arccosh(torch.clamp(-minkowski_ip2(X, X), min=1 + 1e-15)) ** self.p
        )


class DiffusionLorentzHyperbolicAffinity(DiffusionAffinity):
    def __init__(self, beta=1.0, p=2):
        self.beta = beta
        self.p = p
        super(DiffusionLorentzHyperbolicAffinity, self).__init__()

    def compute_log_affinity(self, X):
        return (
            -self.beta * torch.arccosh(torch.clamp(-minkowski_ip2(X, X), min=1 + 1e-15)) ** self.p
        )


class NormalizedLorentzHyperbolicAndStudentAffinity(LogAffinity):
    def __init__(self, student=False, sigma=1.0, gamma=0.1, p=2):
        self.student = student
        self.sigma = sigma
        self.gamma = gamma
        self.p = p
        super(NormalizedLorentzHyperbolicAndStudentAffinity, self).__init__()

    def compute_log_affinity(self, X):
        C = torch.arccosh(torch.clamp(-minkowski_ip2(X, X), min=1 + 1e-15)) ** self.p
        if self.student:
            log_P = -torch.log(math.pi * self.gamma * (1 + C / self.gamma**2))
        else:
            log_P = -C / (2 * self.sigma)
        return log_P - torch.logsumexp(log_P, dim=(0, 1))


class DiffusionGaussianAffinity(DiffusionAffinity):
    def __init__(self, sigma=1.0, p=2, zero_diag=False):
        self.sigma = sigma
        self.p = p
        self.zero_diag = zero_diag
        super(DiffusionGaussianAffinity, self).__init__()

    def compute_log_affinity(self, X):
        return -pairwise_distances(X, X, zero_diag=self.zero_diag) / (2 * self.sigma)


class NormalizedGaussianAndStudentAffinity(LogAffinity):
    """
    This class computes the normalized affinity associated to a Gaussian or t-Student kernel. The affinity matrix is normalized by given axis.

    Parameters
    ----------
    student : bool, optional
        If True computes a t-Student kernel, by default False.
    sigma : float, optional
        The length scale of the Gaussian kernel, by default 1.0.
    p : int, optional
        p value for the p-norm distance to calculate between each vector pair, by default 2.
    """

    def __init__(self, student=False, sigma=1.0, p=2, zero_diag=False):
        self.student = student
        self.sigma = sigma
        self.p = p
        self.zero_diag = zero_diag
        super(NormalizedGaussianAndStudentAffinity, self).__init__()

    def compute_log_affinity(self, X, axis=(0, 1)):
        """
        Computes the pairwise affinity matrix in log space and normalize it by given axis.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """
        C = pairwise_distances(X, X, zero_diag=self.zero_diag)
        if self.student:
            log_P = -torch.log(1 + C)
        else:
            log_P = -C / (2 * self.sigma)
        return log_P - torch.logsumexp(log_P, dim=axis)


class EntropicAffinity(LogAffinity):
    """
    This class computes the entropic affinity used in SNE and tSNE in log domain. It corresponds also to the Pe matrix in [1] in log domain (see also [2]).
    When normalize_as_sne = True, the affinity is symmetrized as (Pe + Pe.T) /2.

    Parameters
    ----------
    perp : int
        Perplexity parameter, related to the number of nearest neighbors that is used in other manifold learning algorithms.
        Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and the number of samples.
        Different values can result in significantly different results. The perplexity must be less than the number of samples.
    tol : _type_, optional
        Precision threshold at which the root finding algorithm stops, by default 1e-5.
    max_iter : int, optional
        Number of maximum iterations for the root finding algorithm, by default 1000.
    verbose : bool, optional
        Verbosity, by default True.
    begin : _type_, optional
        Initial lower bound of the root, by default None.
    end : _type_, optional
        Initial upper bound of the root, by default None.
    normalize_as_sne : bool, optional
        If True the entropic affinity is symmetrized as (Pe + Pe.T) /2, by default True.

    References
    ----------
    [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    [2] Entropic Affinities: Properties and Efficient Numerical Computation, Max Vladymyrov, Miguel A. Carreira-Perpinan, ICML 2013.
    """

    def __init__(
        self,
        perp,
        tol=1e-5,
        max_iter=1000,
        verbose=True,
        begin=None,
        end=None,
        normalize_as_sne=True,
        zero_diag=False,
    ):

        self.perp = perp
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.begin = begin
        self.end = end
        self.normalize_as_sne = normalize_as_sne
        self.zero_diag = zero_diag
        super(EntropicAffinity, self).__init__()

    def compute_log_affinity(self, X):
        """
        Computes the pairwise entropic affinity matrix in log space. If normalize_as_sne is True returns the symmetrized version.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. If normalize_as_sne is True returns the symmetrized affinty in log space.
        """
        C = pairwise_distances(X, X, zero_diag=self.zero_diag)
        log_P = self._solve_dual(C)
        if self.normalize_as_sne:  # does P+P.T/2 in log space
            log_P_SNE = torch.logsumexp(
                torch.stack([log_P, log_P.T], 0), 0, keepdim=False
            ) - math.log(2)
            return log_P_SNE
        else:
            return log_P

    def _solve_dual(self, C):
        """
        Performs a binary search to solve the dual problem of entropic affinities in log space.
        It solves the problem (EA) in [1] and returns the entropic affinity matrix in log space (which is **not** symmetric).

        Parameters
        ----------
        C: torch.Tensor of shape (n_samples, n_samples)
            Distance matrix between the samples.

        References
        ----------
        [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        target_entropy = math.log(self.perp) + 1
        n = C.shape[0]

        if not 1 <= self.perp <= n:
            BadPerplexity("The perplexity parameter must be between 1 and number of samples")

        def f(eps):
            return entropy(log_Pe(C, eps), log=True) - target_entropy

        eps_star, _, _ = false_position(
            f=f,
            n=n,
            begin=self.begin,
            end=self.end,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            device=C.device,
        )
        log_affinity = log_Pe(C, eps_star)

        return log_affinity


class SymmetricEntropicAffinity(LogAffinity):
    def __init__(
        self,
        perp,
        lr=1e0,
        square_parametrization=False,
        tol=1e-3,
        max_iter=500,
        optimizer="Adam",
        verbose=True,
        tolog=False,
        zero_diag=False,
    ):
        """
        This class computes the solution to the symmetric entropic affinity problem described in [1], in log space.
        More precisely, it solves equation (SEA) in [1] with the dual ascent procedure described in the paper and returns the log of the affinity matrix.
        When square_parametrization=False, the problem is convex.

        Parameters
        ----------
        perp : int
            Perplexity parameter, related to the number of nearest neighbors that is used in other manifold learning algorithms.
            Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and the number of samples.
            Different values can result in significantly different results. The perplexity must be less than the number of samples.
        lr : float, optional
            Learning rate for the algorithm, usually in the range [1e-5, 10], by default 1e-3.
        square_parametrization : bool, optional
            Whether to optimize on the square of the dual variables.
            If True the algorithm is not convex anymore but may be more stable in practice, by default False.
        tol : float, optional
            Precision threshold at which the algorithm stops, by default 1e-5.
        max_iter : int, optional
            Number of maximum iterations for the algorithm, by default 500.
        optimizer : {'SGD', 'Adam', 'NAdam'}, optional
            Which pytorch optimizer to use, by default 'Adam'.
        verbose : bool, optional
            Verbosity, by default True.
        tolog : bool, optional
            Whether to store intermediate result in a dictionary, by default False.

        Attributes
        ----------
        log_ : dictionary
            Contains the loss and the dual variables at each iteration of the optimization algorithm when tolog = True.
        n_iter_: int
            Number of iterations run.
        eps_: torch.Tensor of shape (n_samples)
            Dual variable associated to the entropy constraint.
        mu_: torch.Tensor of shape (n_samples)
            Dual variable associated to the marginal constraint.

        References
        ----------
        [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        self.perp = perp
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.verbose = verbose
        self.tolog = tolog
        self.n_iter_ = 0
        self.square_parametrization = square_parametrization
        self.zero_diag = zero_diag
        super(SymmetricEntropicAffinity, self).__init__()

    def compute_log_affinity(self, X):
        """
        Computes the pairwise symmetric entropic affinity matrix in log space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """
        C = pairwise_distances(X, X, zero_diag=self.zero_diag)
        log_P = self._solve_dual(C)
        return log_P

    def _solve_dual(self, C):
        """
        Solves the dual optimization problem (Dual-SEA) in [1] and returns the corresponding symmetric entropic affinty in log space.

        Parameters
        ----------
        C : torch.Tensor of shape (n_samples, n_samples)
            Distance matrix between samples.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.

        References
        ----------
        [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        device = C.device
        n = C.shape[0]
        if not 1 <= self.perp <= n:
            BadPerplexity("The perplexity parameter must be between 1 and number of samples")
        target_entropy = math.log(self.perp) + 1
        # dual variable corresponding to the entropy constraint
        eps = torch.ones(n, dtype=torch.double, device=device)
        # dual variable corresponding to the marginal constraint
        mu = torch.zeros(n, dtype=torch.double, device=device)
        log_P = log_Pse(C, eps, mu, to_square=self.square_parametrization)

        optimizer = OPTIMIZERS[self.optimizer]([eps, mu], lr=self.lr)

        if self.tolog:
            self.log_["eps"] = [eps.clone().detach().cpu()]
            self.log_["mu"] = [mu.clone().detach().cpu()]
            self.log_["loss"] = []

        if self.verbose:
            print("---------- Computing the symmetric entropic affinity matrix ----------")

        one = torch.ones(n, dtype=torch.double, device=device)
        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for k in pbar:
            with torch.no_grad():
                optimizer.zero_grad()
                H = entropy(log_P, log=True)

                if self.square_parametrization:
                    # the Jacobian must be corrected by 2* diag(eps) in the case of square parametrization.
                    eps.grad = 2 * eps.clone().detach() * (H - target_entropy)
                else:
                    eps.grad = H - target_entropy

                P_sum = torch.exp(torch.logsumexp(log_P, -1, keepdim=False))
                mu.grad = P_sum - one
                optimizer.step()
                if not self.square_parametrization:  # optimize on eps > 0
                    eps.clamp_(min=0)

                log_P = log_Pse(C, eps, mu, to_square=self.square_parametrization)

                if torch.isnan(eps).any() or torch.isnan(mu).any():
                    raise NanError(
                        f"NaN in dual variables at iteration {k}, consider decreasing the learning rate of SymmetricEntropicAffinity"
                    )

                if self.tolog:
                    eps0 = eps.clone().detach()
                    mu0 = mu.clone().detach()
                    self.log_["eps"].append(eps0)
                    self.log_["mu"].append(mu0)
                    if self.square_parametrization:
                        self.log_["loss"].append(
                            -Lagrangian(
                                C,
                                torch.exp(log_P.clone().detach()),
                                eps0**2,
                                mu0,
                                self.perp,
                            ).item()
                        )
                    else:
                        self.log_["loss"].append(
                            -Lagrangian(
                                C,
                                torch.exp(log_P.clone().detach()),
                                eps0,
                                mu0,
                                self.perp,
                            ).item()
                        )

                perps = torch.exp(H - 1)
                if self.verbose:
                    pbar.set_description(
                        f"perps : {float(perps.mean().item()): .3e} +-{float(perps.std().item()): .3e}, "
                        f"marginal : {float(P_sum.mean().item()): .3e} +-{float(P_sum.std().item()): .3e}"
                    )

                if (torch.abs(H - math.log(self.perp) - 1) < self.tol).all() and (
                    torch.abs(P_sum - one) < self.tol
                ).all():
                    self.log_["n_iter"] = k
                    self.n_iter_ = k
                    if self.verbose:
                        print(f"breaking at iter {k}")
                    break

                if k == self.max_iter - 1 and self.verbose:
                    print(
                        "---------- Warning: max iter attained, algorithm stops but may not have converged ----------"
                    )

        self.eps_ = eps.clone().detach()
        self.mu_ = mu.clone().detach()

        return log_P


class BistochasticAffinity(LogAffinity):
    """
    This class computes the symmetric doubly stochastic affinity matrix in log domain with Sinkhorn algorithm.
    It normalizes a Gaussian RBF kernel or t-Student kernel to satisfy the doubly stochasticity constraints.

    Parameters
    ----------
    eps : float, optional
        The strength of the regularization for the Sinkhorn algorithm.
        It corresponds to the square root of the length scale of the Gaussian kernel when student = False, by default 1.0.
    f : torch.Tensor of shape (n_samples), optional
        Initialization for the dual variable of the Sinkhorn algorithm, by default None.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-5.
    max_iter : int, optional
        Number of maximum iterations for the algorithm, by default 100.
    student : bool, optional
        Whether to use a t-Student kernel instead of a Gaussian kernel, by default False.
    verbose : bool, optional
        Verbosity, by default False.
    tolog : bool, optional
        Whether to store intermediate result in a dictionary, by default False.

    Attributes
    ----------
    log_ : dictionary
        Contains the dual variables at each iteration of the optimization algorithm when tolog = True.
    n_iter_: int
        Number of iterations run.

    """

    def __init__(
        self,
        eps=1.0,
        f=None,
        tol=1e-5,
        max_iter=100,
        student=False,
        verbose=False,
        tolog=False,
        zero_diag=False,
    ):
        self.eps = eps
        self.f = f
        self.tol = tol
        self.max_iter = max_iter
        self.student = student
        self.tolog = tolog
        self.n_iter_ = 0
        self.verbose = verbose
        self.zero_diag = zero_diag
        super(BistochasticAffinity, self).__init__()

    def compute_log_affinity(self, X):
        """
        Computes the doubly stochastic affinity matrix in log space.
        Returns the log of the transport plan at convergence.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """
        C = pairwise_distances(X, X, zero_diag=self.zero_diag)
        # If student is True, considers the Student-t kernel instead of Gaussian RBF
        if self.student:
            C = torch.log(1 + C)
        log_P = self._solve_dual(C)
        return log_P

    def _solve_dual(self, C):
        """
        Performs Sinkhorn iterations in log domain to solve the entropic "self" (or "symmetric") OT problem with symmetric cost C and entropic regularization eps.

        Parameters
        ----------
        C : torch.Tensor of shape (n_samples, n_samples)
            Distance matrix between samples.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """

        if self.verbose:
            print("---------- Computing the doubly stochastic affinity matrix ----------")
        device = C.device
        n = C.shape[0]

        # Allows a warm-start if a dual variable f is provided
        f = torch.zeros(n, device=device) if self.f is None else self.f

        if self.tolog:
            self.log_["f"] = [f.clone()]

        # Sinkhorn iterations
        for k in range(self.max_iter + 1):
            f = 0.5 * (f - self.eps * torch.logsumexp((f - C) / self.eps, -1))

            if self.tolog:
                self.log_["f"].append(f.clone())

            if torch.isnan(f).any():
                raise NanError(f"NaN in self-Sinkhorn dual variable at iteration {k}")

            log_T = (f[:, None] + f[None, :] - C) / self.eps
            if (torch.abs(torch.exp(torch.logsumexp(log_T, -1)) - 1) < self.tol).all():
                if self.verbose:
                    print(f"breaking at iter {k}")
                break

            if k == self.max_iter - 1 and self.verbose:
                print("---------- max iter attained for Sinkhorn algorithm ----------")

        self.n_iter_ = k

        return (f[:, None] + f[None, :] - C) / self.eps


def log_Pe(C, eps):
    """
    Returns the log of the directed affinity matrix of SNE with prescribed kernel bandwidth.

    Parameters
    ----------
    C : torch.Tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    eps : torch.Tensor of shape (n_samples)
        Kernel bandwidths vector.

    Returns
    -------
    log_P: torch.Tensor of shape (n_samples, n_samples)
        log of the directed affinity matrix of SNE.
    """

    log_P = -C / (eps[:, None])
    return log_P - torch.logsumexp(log_P, -1, keepdim=True)


def log_Pse(C, eps, mu, to_square=False):
    """
    Returns the log of the symmetric entropic affinity matrix with specified parameters epsilon and mu.

    Parameters
    ----------
    C: torch.Tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    eps: torch.Tensor of shape (n_samples)
        Symmetric entropic affinity dual variables associated to the entropy constraint.
    mu: torch.Tensor of shape (n_samples)
        Symmetric entropic affinity dual variables associated to the marginal constraint.
    to_square: bool, optional
        Whether to use the square of the dual variables associated to the entropy constraint, by default False.
    """
    if to_square:
        return (mu[:, None] + mu[None, :] - 2 * C) / (eps[:, None] ** 2 + eps[None, :] ** 2)
    else:
        return (mu[:, None] + mu[None, :] - 2 * C) / (eps[:, None] + eps[None, :])


def Lagrangian(C, log_P, eps, mu, perp):
    """
    Computes the Lagrangian associated to the symmetric entropic affinity optimization problem.

    Parameters
    ----------
    C: torch.Tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    log_P: torch.Tensor of shape (n_samples, n_samples)
        log of the symmetric entropic affinity matrix.
    eps: torch.Tensor of shape (n_samples)
        Dual variable associated to the entropy constraint.
    mu: torch.Tensor of shape (n_samples)
        Dual variable associated to the marginal constraint.
    perp : int
        Perplexity parameter.

    Returns
    -------
    cost: float
        Value of the Lagrangian.
    """
    one = torch.ones(C.shape[0], dtype=torch.double, device=C.device)

    target_entropy = math.log(perp) + 1
    HP = entropy(log_P, log=True, ax=1)
    return (
        torch.exp(torch.logsumexp(log_P + torch.log(C), (0, 1), keepdim=False))
        + torch.inner(eps, (target_entropy - HP))
        + torch.inner(mu, (one - torch.exp(torch.logsumexp(log_P, -1, keepdim=False))))
    )


# %% UMAP affinities

from scipy.optimize import curve_fit
import warnings

from abc import ABC


def symmetric_pairwise_distances(X: torch.Tensor, metric: str, add_diag: float = None):
    r"""Compute pairwise distances matrix between points in a dataset.

    Return the pairwise distance matrix as torch tensor or KeOps lazy tensor
    (if keops is True). Supports batched input. The batch dimension should be the first.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features) or (n_batch, n_samples_batch, n_features)
        Input dataset.
    metric : str, optional
        Metric to use for computing distances. The default is "sqeuclidean".
    keops : bool, optional
        If True, uses KeOps for computing the distances.
    add_diag : float, optional
        If not None, adds weight on the diagonal of the distance matrix.

    Returns
    -------
    C : torch.Tensor or pykeops.torch.LazyTensor (if keops is True) of shape (n_samples, n_samples) or (n_batch, n_samples_batch, n_samples_batch)
        Pairwise distances matrix.
    """  # noqa E501
    C = _pairwise_distances_torch(X, metric=metric)

    if add_diag is not None:  # add mass on the diagonal
        I = torch.eye(C.shape[-1], device=X.device, dtype=X.dtype)
        C += add_diag * I

    return C


def _pairwise_distances_torch(X: torch.Tensor, Y: torch.Tensor = None, metric: str = "sqeuclidean"):
    r"""Compute pairwise distances matrix between points in two datasets.

    Return the pairwise distance matrix as a torch tensor.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        First dataset.
    Y : torch.Tensor of shape (m_samples, n_features)
        Second dataset.
    metric : str
        Metric to use for computing distances.

    Returns
    -------
    C : torch.Tensor of shape (n_samples, m_samples)
        Pairwise distances matrix.
    """

    if Y is None:
        Y = X

    if metric == "sqeuclidean":
        X_norm = (X**2).sum(-1)
        Y_norm = (Y**2).sum(-1)
        C = X_norm.unsqueeze(-1) + Y_norm.unsqueeze(-2) - 2 * X @ Y.transpose(-1, -2)
    elif metric == "euclidean":
        X_norm = (X**2).sum(-1)
        Y_norm = (Y**2).sum(-1)
        C = X_norm.unsqueeze(-1) + Y_norm.unsqueeze(-2) - 2 * X @ Y.transpose(-1, -2)
        C = torch.clip(C, min=0.0).sqrt()  # negative values can appear because of float precision
    elif metric == "manhattan":
        C = (X.unsqueeze(-2) - Y.unsqueeze(-3)).abs().sum(-1)
    elif metric == "angular":
        C = -X @ Y.transpose(-1, -2)
    elif metric == "hyperbolic":
        X_norm = (X**2).sum(-1)
        Y_norm = (Y**2).sum(-1)
        C = (X_norm.unsqueeze(-1) + Y_norm.unsqueeze(-2) - 2 * X @ Y.transpose(-1, -2)) / (
            X[..., 0].unsqueeze(-1) * Y[..., 0].unsqueeze(-2)
        )

    return C


class Affinity_(ABC):
    r"""Base class for affinity matrices.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        The device to use for computation. Typically "cuda" for GPU or "cpu" for CPU.
        If "auto", uses the device of the input data.
    keops : bool, optional
        Whether to use KeOps for efficient computation of large-scale kernel operations.
    verbose : bool, optional
        If True, prints additional information during computation. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        verbose: bool = False,
    ):

        self.log = {}
        self.metric = metric
        self.zero_diag = zero_diag
        self.device = device
        self.verbose = verbose
        self.zero_diag = zero_diag
        self.add_diag = 1e12 if self.zero_diag else None

    def compute_affinity(self, X: torch.Tensor):
        r"""Compute the affinity matrix from the input data.

        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.

        Raises
        ------
        NotImplementedError
            If the `_compute_affinity` method is not implemented by the subclass,
            a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_affinity` method is not implemented."
        )

    def _distance_matrix(self, X: torch.Tensor):
        r"""Compute the pairwise distance matrix from the input data.

        It uses the specified metric and optionally leveraging KeOps
        for memory efficient computation.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : torch.Tensor or pykeops.torch.LazyTensor
            The pairwise distance matrix. The type of the returned matrix depends on the
            value of the `keops` attribute. If `keops` is True, a KeOps LazyTensor
            is returned. Otherwise, a torch.Tensor is returned.
        """
        return symmetric_pairwise_distances(
            X=X,
            metric=self.metric,
            add_diag=self.add_diag,
        )


class LogAffinity_(Affinity_):
    r"""Base class for affinity matrices in log domain.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
    device : str, optional
        The device to use for computation. Typically "cuda" for GPU or "cpu" for CPU.
        If "auto", uses the device of the input data.
    keops : bool, optional
        Whether to use KeOps for efficient computation of large-scale kernel operations.
    verbose : bool, optional
        If True, prints additional information during computation. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            verbose=verbose,
        )

    def compute_affinity(self, X: torch.Tensor, log: bool = False, **kwargs):
        r"""Compute the affinity matrix from the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.
        log : bool, optional
            If True, returns the log of the affinity matrix. Else, returns
            the affinity matrix by exponentiating the log affinity matrix.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed log affinity matrix if `log` is True, otherwise the
            exponentiated log affinity matrix.
        """
        if self.device == "auto":
            self.device = X.device

        if X.device != self.device:
            X = X.to(self.device)

        log_affinity = self._compute_log_affinity(X, **kwargs)
        if log:
            return log_affinity
        else:
            return log_affinity.exp()

    def _compute_log_affinity(self, X: torch.Tensor):
        r"""Compute the log affinity matrix from the input data.

        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.

        Raises
        ------
        NotImplementedError
            If the `_compute_log_affinity` method is not implemented by the subclass,
            a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_log_affinity` method is not implemented."
        )


class SparseLogAffinity(LogAffinity_):
    r"""Base class for sparse log affinity matrices.

    If sparsity is enabled, returns the log affinity matrix in a rectangular format
    with the corresponding indices.
    Otherwise, returns the full affinity matrix and None.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
        Default is "sqeuclidean".
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero. Default is True.
    device : str, optional
        The device to use for computation. Typically "cuda" for GPU or "cpu" for CPU.
        If "auto", uses the device of the input data. Default is "auto".
    keops : bool, optional
        Whether to use KeOps for efficient computation of large-scale kernel
        operations. Default is False.
    verbose : bool, optional
        If True, prints additional information during computation. Default is False.
    sparsity : bool or 'auto', optional
        Whether to compute the affinity matrix in a sparse format. Default is "auto".
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        verbose: bool = False,
        sparsity: str = "auto",
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            verbose=verbose,
        )
        self.sparsity = sparsity
        if sparsity == "auto":
            self._sparsity = self._sparsity_rule()
        else:
            self._sparsity = sparsity

    def _sparsity_rule(self):
        r"""Rule to determine whether to compute the affinity matrix in a sparse format.

        This method must be overridden by subclasses.

        Raises
        ------
        NotImplementedError
            If the `_sparsity_rule` method is not implemented by the subclass,
            a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_sparsity_rule` method is not implemented. "
            "Therefore sparsity = 'auto' is not supported."
        )

    def compute_affinity(
        self,
        X: torch.Tensor,
        log: bool = False,
        return_indices: bool = False,
        **kwargs,
    ):
        r"""Compute and return the log affinity matrix from input data.

        If sparsity is enabled, returns the log affinity in rectangular format with the
        corresponding indices. Otherwise, returns the full affinity matrix and None.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data used to compute the affinity matrix.
        log : bool, optional
            If True, returns the log of the affinity matrix. Else, returns
            the affinity matrix by exponentiating the log affinity matrix.
        return_indices : bool, optional
            If True, returns the indices of the non-zero elements in the affinity matrix
            if sparsity is enabled. Default is False.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed log affinity matrix if `log` is True, otherwise the
            exponentiated log affinity matrix.
        indices : torch.Tensor
            If return_indices is True, returns the indices of the non-zero elements
            in the affinity matrix if sparsity is enabled. Otherwise, returns None.
        """
        if self.device == "auto":
            self.device = X.device

        if X.device != self.device:
            X = X.to(self.device)

        log_affinity, indices = self._compute_sparse_log_affinity(X, **kwargs)
        affinity_to_return = log_affinity if log else log_affinity.exp()
        return (affinity_to_return, indices) if return_indices else affinity_to_return

    def _compute_sparse_log_affinity(self, X: torch.Tensor):
        r"""Compute the log affinity matrix in a sparse format from the input data.

        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.

        Raises
        ------
        NotImplementedError
            If the `_compute_sparse_log_affinity` method is not implemented by
            the subclass, a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_sparse_log_affinity` method is " "not implemented."
        )


def _log_Pumap(C, rho, sigma):
    r"""Return the log of the input affinity matrix used in UMAP."""
    return -(C - rho[:, None]) / sigma[:, None]


# from umap/umap/umap_.py
def find_ab_params(spread, min_dist):
    """Fit a, b params as in UMAP.

    Fit (a, b) for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def _check_n_neighbors(n_neighbors, n, verbose=True):
    r"""Check the n_neighbors parameter and returns a valid value."""
    if n <= 1:
        raise ValueError(f"[TorchDR] ERROR : Input has less than one sample : n_samples = {n}.")

    if n_neighbors >= n or n_neighbors <= 1:
        new_value = n // 2
        if verbose:
            warnings.warn(
                "[TorchDR] WARNING : The n_neighbors parameter must be greater than "
                f"1 and smaller than the number of samples (here n = {n}). "
                f"Got n_neighbors = {n_neighbors}. Setting n_neighbors to {new_value}."
            )
        return new_value
    else:
        return n_neighbors


class UMAPAffinityIn(SparseLogAffinity):
    r"""Compute the input affinity used in UMAP [8]_.

    The algorithm computes via root search the variable
    :math:`\mathbf{\sigma}^* \in \mathbb{R}^n_{>0}` such that

    .. math::
        \forall i, \: \sum_j P_{ij} = \log (\mathrm{n_neighbors}) \quad \text{where} \quad \forall (i,j), \: P_{ij} = \exp(- (C_{ij} - \rho_i) / \sigma^\star_i)

    and :math:`\rho_i = \min_j C_{ij}`.

    Parameters
    ----------
    n_neighbors : float, optional
        Number of effective nearest neighbors to consider. Similar to the perplexity.
    tol : float, optional
        Precision threshold for the root search.
    max_iter : int, optional
        Maximum number of iterations for the root search.
    sparsity : bool or 'auto', optional
        Whether to use sparsity mode.
    metric : str, optional
        Metric to use for pairwise distances computation.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        Device to use for computations.
    verbose : bool, optional
        Verbosity. Default is False.

    References
    ----------
    .. [8] Leland McInnes, John Healy, James Melville (2018).
        UMAP: Uniform manifold approximation and projection for dimension reduction.
        arXiv preprint arXiv:1802.03426.

    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 30,  # analog of the perplexity parameter of SNE / TSNE
        tol: float = 1e-5,
        max_iter: int = 1000,
        sparsity: str = "auto",
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        verbose: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.tol = tol
        self.max_iter = max_iter

        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            verbose=verbose,
            sparsity=sparsity,
        )

    def _sparsity_rule(self):
        """
        if self.n_neighbors < 100:
            return True
        else:
            if self.verbose:
                warnings.warn(
                    "[TorchDR] WARNING Affinity: n_neighbors is large "
                    f"({self.n_neighbors}) thus we turn off sparsity for "
                    "the EntropicAffinity. "
                )
            return False
        """
        return False

    def _compute_sparse_log_affinity(self, X: torch.Tensor):
        r"""Compute the input affinity matrix of UMAP from input data X.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : UMAPAffinityData
            The fitted instance.
        """
        if self.verbose:
            print("[TorchDR] Affinity : computing the input affinity matrix of UMAP.")

        C = self._distance_matrix(X)

        n_samples_in = C.shape[0]
        n_neighbors = _check_n_neighbors(self.n_neighbors, n_samples_in, self.verbose)

        if self._sparsity:
            if self.verbose:
                print("[TorchDR] Affinity : sparsity mode enabled, computing " "nearest neighbors.")
            # when using sparsity, we construct a reduced distance matrix
            # of shape (n_samples, n_neighbors)
            C_, indices = C.topk(k=n_neighbors, dim=1, largest=False)
            indices = indices.int()

        else:
            C_, indices = C, None

        self.rho_ = C_.topk(k=1, dim=1, largest=False)[0].squeeze().contiguous()

        def marginal_gap(eps):  # function to find the root of
            marg = _log_Pumap(C_, self.rho_, eps).logsumexp(1).exp().squeeze()
            return marg - math.log(n_neighbors)

        self.eps_, _, _ = false_position(
            f=marginal_gap,
            n=n_samples_in,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            device=X.device,
        )

        log_affinity_matrix = _log_Pumap(C_, self.rho_, self.eps_)

        return log_affinity_matrix, indices


# %%


def pairwise_distances(
    X: torch.Tensor,
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    zero_diag: bool = False,
):
    r"""Compute pairwise distances matrix between points in two datasets.

    Returns the pairwise distance matrix as torch tensor or KeOps lazy tensor
    (if keops is True).

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        First dataset.
    Y : torch.Tensor of shape (m_samples, n_features), optional
        Second dataset. If None, Y = X.
    metric : str, optional
        Metric to use for computing distances. The default is "sqeuclidean".
    keops : bool, optional
        If True, uses KeOps for computing the distances.

    Returns
    -------
    C : torch.Tensor or pykeops.torch.LazyTensor (if keops is True)
    of shape (n_samples, m_samples)
        Pairwise distances matrix.
    """
    if Y is None:
        Y = X

    C = _pairwise_distances_torch(X, Y, metric)

    if zero_diag:
        I = torch.eye(C.shape[-1], device=X.device, dtype=X.dtype)
        C += 1e12 * I

    return C


def symmetric_pairwise_distances_indices(
    X: torch.Tensor,
    indices: torch.Tensor,
    metric: str = "sqeuclidean",
):
    r"""Compute pairwise distances for a subset of pairs given by indices.

    The output distance matrix has shape (n, k) and its (i,j) element is the
    distance between X[i] and Y[indices[i, j]].

    Parameters
    ----------
    X : torch.Tensor of shape (n, p)
        Input dataset.
    indices : torch.Tensor of shape (n, k)
        Indices of the pairs for which to compute the distances.
    metric : str, optional
        Metric to use for computing distances. The default is "sqeuclidean".

    Returns
    -------
    C_indices : torch.Tensor of shape (n, k)
        Pairwise distances matrix for the subset of pairs.
    """
    X_indices = X[indices.int()]  # Shape (n, k, p)

    if metric == "sqeuclidean":
        C_indices = torch.sum((X.unsqueeze(1) - X_indices) ** 2, dim=-1)
    elif metric == "euclidean":
        C_indices = torch.sum((X.unsqueeze(1) - X_indices) ** 2, dim=-1).sqrt()
    elif metric == "manhattan":
        C_indices = torch.sum(torch.abs(X.unsqueeze(1) - X_indices), dim=-1)
    elif metric == "angular":
        C_indices = -torch.sum(X.unsqueeze(1) * X_indices, dim=-1)
    elif metric == "hyperbolic":
        C_indices = torch.sum((X.unsqueeze(1) - X_indices) ** 2, dim=-1) / (
            X[:, 0].unsqueeze(1) * X_indices[:, :, 0]
        )
    else:
        raise NotImplementedError(f"Metric '{metric}' is not (yet) implemented.")

    return C_indices


class UnnormalizedAffinity_(Affinity_):
    r"""Base class for unnormalized affinities.

    These affinities are defined using a closed-form formula on the pairwise distance
    matrix and can be directly applied to a subset of the data by providing indices.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
        Default is "sqeuclidean".
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero. Default is True.
    device : str, optional
        The device to use for computation, e.g., "cuda" for GPU or "cpu" for CPU.
        If "auto", it uses the device of the input data. Default is "auto".
    keops : bool, optional
        Whether to use KeOps for efficient computation of large-scale kernel
        operations. Default is False.
    verbose : bool, optional
        If True, prints additional information during computation. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            verbose=verbose,
        )

    def compute_affinity(
        self,
        X: torch.Tensor,
        Y: torch.Tensor = None,
        indices: torch.Tensor = None,
        **kwargs,
    ):
        r"""Compute the affinity matrix from the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second input data. If None, uses `Y=X`. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. If None, computes the full affinity matrix.
            Default is None.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix.
        """
        if self.device == "auto":
            self.device = X.device

        if X.device != self.device:
            X = X.to(self.device)

        if Y is not None:
            if Y.device != self.device:
                Y = Y.to(self.device)

        C = self._distance_matrix(X=X, Y=Y, indices=indices, **kwargs)
        return self._affinity_formula(C)

    def _affinity_formula(self, C: torch.Tensor):
        r"""Compute the affinity from the distance matrix.

        This method must be overridden by subclasses.

        Parameters
        ----------
        C : torch.Tensor or pykeops.torch.LazyTensor
            Pairwise distance matrix.

        Raises
        ------
        NotImplementedError
            If the `_affinity_formula` method is not implemented by the subclass,
            a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_affinity_formula` method is not implemented."
        )

    def _distance_matrix(
        self,
        X: torch.Tensor,
        Y: torch.Tensor = None,
        indices: torch.Tensor = None,
    ):
        r"""Compute the pairwise distance matrix from the input data.

        It uses the specified metric and optionally leverages KeOps
        for memory efficient computation.
        It supports computing the full pairwise distance matrix, the pairwise
        distance matrix between two sets of samples, or the pairwise distance matrix
        between a set of samples and a subset of samples specified by indices.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second input data. If None, uses `Y=X`. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. If None, computes the full pairwise distance
            matrix. Default is None.

        Returns
        -------
        C : torch.Tensor or pykeops.torch.LazyTensor
            The pairwise distance matrix. The type of the returned matrix depends on the
            value of the `keops` attribute. If `keops` is True, a KeOps LazyTensor
            is returned. Otherwise, a torch.Tensor is returned.
        """
        if Y is not None and indices is not None:
            raise NotImplementedError(
                "[TorchDR] ERROR : transform method cannot be called with both Y "
                "and indices at the same time."
            )

        elif indices is not None:
            return symmetric_pairwise_distances_indices(X, indices=indices, metric=self.metric)

        elif Y is not None:
            return pairwise_distances(X, Y, metric=self.metric)

        else:
            return symmetric_pairwise_distances(X, metric=self.metric, add_diag=self.add_diag)


class UMAPAffinityOut(UnnormalizedAffinity_):
    r"""Compute the affinity used in embedding space in UMAP [8]_.

    Its :math:`(i,j)` coefficient is as follows:

    .. math::
        1 / \left(1 + a C_{ij}^{b} \right)

    where parameters a and b are fitted to the spread and min_dist parameters.

    Parameters
    ----------
    min_dist : float, optional
        min_dist parameter from UMAP. Provides the minimum distance apart that
        points are allowed to be.
    spread : float, optional
        spread parameter from UMAP.
    a : float, optional
        factor of the cost matrix.
    b : float, optional
        exponent of the cost matrix.
    degrees_of_freedom : int, optional
        Degrees of freedom for the Student-t distribution.
    metric : str, optional
        Metric to use for pairwise distances computation.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity. Default is False.

    References
    ----------
    .. [8] Leland McInnes, John Healy, James Melville (2018).
        UMAP: Uniform manifold approximation and projection for dimension reduction.
        arXiv preprint arXiv:1802.03426.
    """

    def __init__(
        self,
        min_dist: float = 0.1,
        spread: float = 1,
        a: float = None,
        b: float = None,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            verbose=verbose,
        )
        self.min_dist = min_dist
        self.spread = spread

        if a is None or b is None:
            fitted_a, fitted_b = find_ab_params(self.spread, self.min_dist)
            self._a, self._b = fitted_a.item(), fitted_b.item()
        else:
            self._a = a
            self._b = b

    def _affinity_formula(self, C: torch.Tensor):
        return 1 / (1 + self._a * C**self._b)
