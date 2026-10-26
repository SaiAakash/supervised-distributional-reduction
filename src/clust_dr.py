"""
Contains classes to run all methods involved in the benchmark, namely:
    - Our method Distributional Reduction DistR
    - COOTClust
    - Clust_then_DR
    - DR_then_Clust
"""

from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from abc import abstractmethod

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

from ot.gromov import gwloss, init_matrix
from local_ot.gromov._utils import init_matrix_semirelaxed
from local_ot.gromov._semirelaxed import (
    semirelaxed_gromov_wasserstein,
    entropic_semirelaxed_gromov_wasserstein,
)
from local_ot.fused_gromov._semirelaxed import (
    semirelaxed_fused_gromov_wasserstein,
    entropic_semirelaxed_fused_gromov_wasserstein,
)

from src.affinities import GramAffinity

from ot.backend import get_backend
from ot.coot import co_optimal_transport

from src.affinities import (
    BaseAffinity,
    LogAffinity,
    NanError,
    UnnormalizedAffinity_,
    SparseLogAffinity,
)
from src.hsic_loss import hsic

from src.utils import (
    barycenter_feat,
    KMeans,
)  # GPU friendly version of kmeans used for spectral clustering
from src.utils_hyperbolic import sampleLorentzNormal
from src.dimension_reduction import AffinityMatcher

import geoopt


class NotBaseAffinityError(Exception):
    pass


class WrongInputFitError(Exception):
    pass


class WrongParameter(Exception):
    pass


OPTIMIZERS = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "NAdam": torch.optim.NAdam,
    "RAdam": geoopt.optim.RiemannianAdam,
}


class DataSummarizer:
    """
    This class groups methods producing a compressed representation of the data in both column (DR) and row (clustering) axis.
    Given a dataset X, an integer n and a dimension d, it computes a n x d representation of the input.

    Parameters
    ----------
    output_sam : int, optional
        Number of rows of the embedding matrix, by default 10.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use, by default 'Adam'.
    lr : float, optional
        Learning rate for the (embedding) algorithm, by default 1.0.
    init : {'random'} or torch.Tensor of shape (output_sam, output_dim), optional
        Initialization for the embedding Z, default 'random'.
    verbose : bool, optional
        Verbosity, by default True.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-6.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    seed : int, optional
        Random seed, by default 0.
    device: str or torch.device type, optional
        Device to perform computations, by default 'cpu'.
    dtype: torch.dtype, optional
        dtype used throughout computations, by defaut torch.double.
    """

    def __init__(
        self,
        output_sam=10,
        output_dim=2,
        optimizer="Adam",
        lr=1.0,
        init="normal",
        verbose=True,
        tol=1e-6,
        max_iter=1000,
        seed=0,
        device="cpu",
        dtype=torch.double,
    ):

        self.output_sam = output_sam
        self.output_dim = output_dim
        assert optimizer in OPTIMIZERS.keys()
        self.optimizer = optimizer
        self.lr = lr
        self.init = init
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.seed = seed
        self.device = device
        self.generator = torch.Generator(self.device)  # Always use CPU generator
        self.generator.manual_seed(seed)
        self.dtype = dtype
        self.losses = []
        self._init_embedding()

    def fit(self, X, y=None):
        """
        Fit X into an embedded space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_data="precomputed".
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        self.fit_transform(X)
        return self

    @abstractmethod
    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and returns the embedding.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_embedding="precomputed".
        y : None
            Ignored.

        Returns
        -------
        Z : torch.Tensor of shape (output_sam, output_dim)
            Embedding of the training data in low-dimensional space.
        """
        pass

    def _update_embedding(self):
        """
        Optimize the embeddings coordinates using a gradient-based optimization method.
        """
        self.Z.requires_grad = True
        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for i in pbar:
            self.optimizer.zero_grad()
            Loss = self._embed_loss()
            if torch.isnan(Loss):
                raise NanError("NaN in embedding loss")
            Loss.backward()
            self.optimizer.step()

            self.losses.append(Loss.item())
            if i > 1:
                delta = abs(self.losses[-1] - self.losses[-2]) / abs(self.losses[-2])
                if delta < self.tol:
                    if self.verbose:
                        print("---------- delta loss convergence ----------")
                    break
                if self.verbose:
                    pbar.set_description(
                        f"Loss : {float(self.losses[-1]): .3e}, " f"delta : {float(delta): .3e} "
                    )

    def _init_embedding(self):
        """
        Initialize the embeddings, either with given initial coordinates if init is a tensor or by sampling i.i.d. Gaussian variables.
        """
        if isinstance(self.init, torch.Tensor):
            if self.init.shape[0] != self.output_sam or self.init.shape[1] != self.output_dim:
                raise WrongParameter(
                    "Init tensor must be of shape {0} but found {1}".format(
                        (self.output_sam, self.output_dim), self.init.shape
                    )
                )
            self.Z = self.init.to(self.device)

        elif self.init == "normal":
            self.Z = torch.normal(
                0,
                1,
                size=(self.output_sam, self.output_dim),
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
        elif self.init == "WrappedNormal":
            self.Z = geoopt.ManifoldTensor(
                sampleLorentzNormal(self.output_sam, self.output_dim + 1, seed=self.seed).double(),
                manifold=geoopt.Lorentz(),
            ).to(dtype=self.dtype, device=self.device)
        else:
            raise WrongParameter('init must be in ["normal", "WrappedNormal"]')
        self.optimizer = OPTIMIZERS[self.optimizer]([self.Z], lr=self.lr)

    @abstractmethod
    def _embed_loss(self):
        """
        Embedding loss function. Must be overridden.
        """
        pass


class COOTClust(DataSummarizer):
    """
    This class solves the joint Clust-DR problem using Co - Optimal Transport (COOT) [1] producing a compressed representation of the data (in both row and column axis).
    Given a dataset X, an integer n and a dimension d, it computes a n x d representation of the input.

    Parameters
    ----------
    output_sam : int, optional
        Number of rows of the embedding matrix, by default 10.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use, by default 'Adam'.
    lr : float, optional
        Learning rate for the (embedding) algorithm, by default 1.0.
    init : {'random'} or torch.Tensor of shape (output_sam, output_dim), optional
        Initialization for the embedding Z, default 'random'.
    verbose : bool, optional
        Verbosity, by default True.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-6.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    seed : int, optional
        Random seed, by default 0.
    device: str or torch.device type, optional
        Device to perform computations, by default 'cpu'.
    dtype: torch.dtype, optional
        dtype used throughout computations, by defaut torch.double.
    max_iter_outer : int, optional
        Number of outer BCD iterations, by default 10.

    [1] Ievgen Redko, Titouan Vayer, Rémi Flamary, and Nicolas Courty, CO-Optimal Transport,
        Advances in Neural Information Processing ny_sampstems, 33 (2020).

    """

    def __init__(
        self,
        output_sam=10,
        output_dim=2,
        optimizer="Adam",
        lr=1.0,
        init="random",
        verbose=True,
        tol=1e-6,
        max_iter=1000,
        seed=0,
        device="cpu",
        dtype=torch.double,
        max_iter_outer=10,
    ):

        super(COOTClust, self).__init__(
            output_sam=output_sam,
            output_dim=output_dim,
            optimizer=optimizer,
            lr=lr,
            init=init,
            verbose=verbose,
            tol=tol,
            max_iter=max_iter,
            seed=seed,
            device=device,
            dtype=dtype,
        )

        self.max_iter_outer = max_iter_outer
        self.running_loss = torch.inf
        self.running_Z = None

    def _update_T(self):
        """
        Compute Co-Optimal Transport plans with fixed Z.
        """
        self.T, self.T_feat, log = co_optimal_transport(
            self.X, self.Z, warmstart=self.init_T, log=True
        )
        self.init_T = {
            "duals_sample": log["duals_sample"],
            "duals_feature": log["duals_feature"],
            "pi_sample": self.T,
            "pi_feature": self.T_feat,
        }

    def _embed_loss(self):
        """
        CO-Optimal Transport loss function.
        """
        XZ_sqr_T = ((self.X.T) ** 2 @ self.wx_samp)[:, None] + ((self.Z.T) ** 2 @ self.wz_samp)[
            None, :
        ]
        ot_cost = XZ_sqr_T - 2 * self.X.T @ self.T @ self.Z
        Loss = (ot_cost * self.T_feat).sum()
        return Loss

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and returns the embedding.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_embedding="precomputed".
        y : None
            Ignored.

        Returns
        -------
        Z : torch.Tensor of shape (output_sam, output_dim)
            Embedding of the training data in low-dimensional space.
        """

        self.X = X
        self.wx_samp = torch.ones(X.shape[0], device=self.device, dtype=self.dtype) / X.shape[0]
        self.wz_samp = (
            torch.ones(self.output_sam, device=self.device, dtype=self.dtype) / self.output_sam
        )
        self.init_T = None

        for t in range(self.max_iter_outer):

            if self.verbose:
                print(f"--- {t}-th outer loop ---")

            with torch.no_grad():
                self._update_T()
            self._update_embedding()

            embed_loss = self._embed_loss()
            if embed_loss > self.running_loss:
                self.Z = self.running_Z
                if self.verbose:
                    print(f"--- Breaking : BCD iteration increasing the loss. ---")
                break
            else:
                self.running_loss = embed_loss
                self.running_Z = self.Z

        self.Z = self.Z.detach()
        return self.Z


class AffinityBasedDataSummarizer(DataSummarizer):
    """
    This class groups affinity based methods producing a compressed representation of the data in both column (DR) and row (clustering) axis.
    Given a dataset X, an integer n and a dimension d, it computes a n x d representation of the input.

    Parameters
    ----------
    affinity_data : "precomputed" or BaseAffinity
        The affinity in the input space (in X, corresponds to P_X).
        If affinity_data is "precomputed" then a affinity matrix (instead of a BaseAffinity object) is needed as input for the fit method.
    affinity_embedding : BaseAffinity
        The affinity in the embedding space (in Z, corresponds to Q_Z).
    output_sam : int, optional
        Number of rows of the embedding matrix, by default 10.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use, by default 'Adam'.
    lr : float, optional
        Learning rate for the (embedding) algorithm, by default 1.0.
    init : {'random'} or torch.Tensor of shape (output_sam, output_dim), optional
        Initialization for the embedding Z, default 'random'.
    verbose : bool, optional
        Verbosity, by default True.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-6.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    seed : int, optional
        Random seed, by default 0.
    device: str or torch.device type, optional
        Device to perform computations, by default 'cpu'.
    dtype: torch.dtype, optional
        dtype used throughout computations, by defaut torch.double.
    loss_fun : {'kl_loss', 'kl_nomarg_loss', 'square_loss'}, optional
        Loss used in the DR problem, by default 'square_loss'.
    """

    def __init__(
        self,
        affinity_data=GramAffinity(centering=True),
        affinity_embedding=GramAffinity(),
        output_sam=10,
        output_dim=2,
        optimizer="Adam",
        lr=1.0,
        init="random",
        verbose=True,
        tol=1e-6,
        max_iter=1000,
        seed=0,
        device="cpu",
        dtype=torch.double,
        loss_fun="square_loss",
        init_T="spectral",
    ):

        super(AffinityBasedDataSummarizer, self).__init__(
            output_sam=output_sam,
            output_dim=output_dim,
            optimizer=optimizer,
            lr=lr,
            init=init,
            verbose=verbose,
            tol=tol,
            max_iter=max_iter,
            seed=seed,
            device=device,
            dtype=dtype,
        )

        if (
            not isinstance(affinity_data, (BaseAffinity, SparseLogAffinity))
            and not affinity_data == "precomputed"
        ):
            raise NotBaseAffinityError('affinity_data  must be BaseAffinity or "precomputed".')
        self.affinity_data = affinity_data

        if not isinstance(affinity_embedding, (BaseAffinity, UnnormalizedAffinity_)):
            raise NotBaseAffinityError(
                "affinity_embedding must be BaseAffinity and implement a compute_affinity method."
            )
        assert loss_fun in ["square_loss", "kl_loss", "kl_nomarg_loss", "binary_cross_entropy"]
        self.loss_fun = loss_fun

        if self.loss_fun == "kl_loss" or self.loss_fun == "kl_nomarg_loss":
            if not isinstance(affinity_embedding, LogAffinity):
                raise NotBaseAffinityError(
                    'affinity_embedding  must be LogAffinity when loss_fun is "kl_loss" or "kl_nomarg_loss".'
                )
        self.affinity_embedding = affinity_embedding

        self.init_T = init_T

    def _compute_affinity(self, X):
        """
        Computes the input affinity matrix or retrieves it when affinity_data="precomputed".

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_data="precomputed".
        """
        if isinstance(X, torch.Tensor):
            self.X = X
        else:
            raise WrongInputFitError(
                'X must be a tensor : either the input data or the affinity when affinity_data="precomputed".'
            )

        if isinstance(self.affinity_data, (BaseAffinity, SparseLogAffinity)):
            self.PX = self.affinity_data.compute_affinity(self.X)
        elif self.affinity_data == "precomputed":
            if self.X.shape[0] != self.X.shape[1]:
                raise WrongInputFitError(
                    'When affinity_data="precomputed" the input X in fit must be a torch.Tensor of shape (n_samples, n_samples)'
                )
            if not torch.all(X >= 0):
                raise WrongInputFitError(
                    'When affinity_data="precomputed" the input X in fit must be non-negative'
                )
            self.PX = X
        else:
            raise WrongParameter("Affinity data not implemented")


class DistR(AffinityBasedDataSummarizer):
    """
    This class solves the joint Clust-DR problem using Gromov-Wasserstein producing a compressed representation of the data (in both row and column axis).
    If alpha < 1 is provided, the algorithm uses fused Gromov-Wasserstein [2].
    Given a dataset X, an integer n and a dimension d, it computes a n x d representation of the input.

    Parameters
    ----------
    affinity_data : "precomputed" or BaseAffinity
        The affinity in the input space (in X, corresponds to P_X).
        If affinity_data is "precomputed" then a affinity matrix (instead of a BaseAffinity object) is needed as input for the fit method.
    affinity_embedding : BaseAffinity
        The affinity in the embedding space (in Z, corresponds to Q_Z).
    output_sam : int, optional
        Number of rows of the embedding matrix, by default 10.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    loss_fun : {'kl_loss', 'kl_nomarg_loss' 'square_loss'}, optional
        Loss used in the DR problem, by default 'square_loss'.
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use, by default 'Adam'.
    lr : float, optional
        Learning rate for the (embedding) algorithm, by default 1.0.
    init : {'random'} or torch.Tensor of shape (output_sam, output_dim), optional
        Initialization for the embedding Z, default 'random'.
    verbose : bool, optional
        Verbosity, by default True.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-6.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    seed : int, optional
        Random seed, by default 0.
    device: str or torch.device type, optional
        Device to perform computations, by default 'cpu'.
    dtype: torch.dtype, optional
        dtype used throughout computations, by defaut torch.double.
    alpha : float, optional
        If alpha < 1, computed fused srGW with interpolation parameter alpha, by default 1.
    init_T : {'random', 'unif', 'kmeans', 'softkmeans'}, optional
        Initialization for the transport plan T, by default 'kmeans'.
    max_iter_outer : int, optional
        Number of outer BCD iterations, by default 10.
    marginal_loss: bool, optional
        Either to take into account the constant marginal terms or not in the GW loss, by default False.

    [2] Titouan Vayer, Laetitia Chapel, Rémi Flamary, Romain Tavenard and Nicolas Courty
    "Optimal Transport for structured data with application on graphs" International Conference on Machine Learning (ICML). 2019.

    """

    def __init__(
        self,
        affinity_data=GramAffinity(centering=True),
        affinity_embedding=GramAffinity(),
        output_sam=10,
        output_dim=2,
        loss_fun="square_loss",
        optimizer="Adam",
        lr=1.0,
        init="normal",  # Bug fixed here, this used to be "random" but there is no random init strategy in the code
        verbose=True,
        tol=1e-6,
        max_iter=1000,
        seed=0,
        device="cpu",
        dtype=torch.double,
        init_T="spectral",
        alpha=1,
        max_iter_outer=10,
        marginal_loss=False,
        entropic_reg=0.0,
        early_stopping=True,
    ):

        super(DistR, self).__init__(
            affinity_data=affinity_data,
            affinity_embedding=affinity_embedding,
            output_sam=output_sam,
            output_dim=output_dim,
            loss_fun=loss_fun,
            optimizer=optimizer,
            lr=lr,
            init=init,
            verbose=verbose,
            tol=tol,
            max_iter=max_iter,
            seed=seed,
            device=device,
            dtype=dtype,
            init_T=init_T,
        )

        self.max_iter_outer = max_iter_outer
        self.alpha = alpha
        self.marginal_loss = marginal_loss
        self.entropic_reg = entropic_reg
        self.running_loss = torch.inf
        self.running_Z = None
        self.early_stopping = early_stopping
        if self.affinity_data == "precomputed":
            if self.alpha < 1:
                raise WrongParameter(
                    'When affinity_data="precomputed" the fused GW alpha parameter must be set to 1 (no linear term).'
                )

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and returns the embedding.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_embedding="precomputed".
        y : None
            Ignored.

        Returns
        -------
        Z : torch.Tensor of shape (output_sam, output_dim)
            Embedding of the training data in low-dimensional space.
        """
        self.backend = get_backend(X)
        self.X = X

        self._compute_affinity(X)

        self._init_T()

        for t in range(self.max_iter_outer):

            if self.verbose:
                print(f"--- {t}-th outer loop ---")

            self._update_embedding()

            with torch.no_grad():
                self._update_T()

            embed_loss = self._embed_loss()
            if (embed_loss > self.running_loss) and self.early_stopping:
                self.Z = self.running_Z
                if self.verbose:
                    print(f"--- Breaking : BCD iteration increasing the loss ---")
                break
            else:
                self.running_loss = embed_loss
                self.running_Z = self.Z

        self.Z = self.Z.detach()
        return self.Z

    def _init_T(self):
        """
        Initializes the Gromov-Wasserstein optimal transport plan.
        """
        self.N = self.PX.shape[0]
        self.h0 = torch.ones(self.N, dtype=self.dtype, device=self.device)
        if isinstance(self.init_T, torch.Tensor):
            if self.init_T.shape[0] != self.N or self.init_T.shape[1] != self.output_sam:
                raise WrongParameter(
                    "init_T tensor must be of shape {0} but found {1}".format(
                        (self.N, self.output_sam), self.init_T.shape
                    )
                )
            self.T = self.init_T.to(dtype=self.dtype, device=self.device)
        elif self.init_T == "random":
            T0 = torch.rand(
                (self.N, self.output_sam),
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
            self.T = T0 / T0.sum(-1, keepdim=True)
        elif self.init_T == "unif":
            q = torch.ones(self.output_sam, dtype=self.dtype, device=self.device) / self.output_sam
            self.T = torch.outer(self.h0, q)
        elif self.init_T == "kmeans":
            kmeans = KMeans(n_clusters=self.output_sam, random_state=self.seed, n_init=10).fit(
                self.X.cpu()
            )  # Apply kmeans on data
            self.T = torch.eye(self.output_sam)[kmeans.labels_].to(
                dtype=self.dtype, device=self.device
            )
        elif self.init_T == "spectral":
            KX = self.PX
            if (
                KX < 0
            ).any():  # Ensuring KX has positive coefficients for performing spectral clustering
                KX -= KX.min()
            kmeans = SpectralClustering(
                n_clusters=self.output_sam,
                random_state=self.seed,
                affinity="precomputed",
                n_init=10,
            ).fit(
                KX.cpu()
            )  # Apply spectral clustering on affinity matrix
            self.T = torch.eye(self.output_sam)[kmeans.labels_].to(
                dtype=self.dtype, device=self.device
            )
        elif self.init_T == "softkmeans":
            kmeans = KMeans(n_clusters=self.output_sam, random_state=self.seed, n_init=10).fit(
                self.X.cpu()
            )  # Apply kmeans on data
            self.T = torch.eye(self.output_sam)[kmeans.labels_].to(
                dtype=self.dtype, device=self.device
            )
            q = torch.ones(self.output_sam, dtype=self.dtype, device=self.device) / self.output_sam
            self.T = (self.T + torch.outer(self.h0, q)) / 2.0

        elif self.init_T == "softspectral":
            KX = self.PX
            if (
                KX < 0
            ).any():  # Ensuring KX has positive coefficients for performing spectral clustering
                KX -= KX.min()
            kmeans = SpectralClustering(
                n_clusters=self.output_sam,
                random_state=self.seed,
                affinity="precomputed",
                n_init=10,
            ).fit(
                KX.cpu()
            )  # Apply spectral clustering on affinity matrix
            self.T = torch.eye(self.output_sam)[kmeans.labels_].to(
                dtype=self.dtype, device=self.device
            )
            q = torch.ones(self.output_sam, dtype=self.dtype, device=self.device) / self.output_sam
            self.T = (self.T + torch.outer(self.h0, q)) / 2.0
        else:
            raise WrongParameter(
                'init_T must be in ["random", "unif", "kmeans", "spectral", "softkmeans"] or be a tensor.'
            )

    def _embed_loss(self):
        """
        Gromov-Wasserstein loss function.
        """
        PZ = self.affinity_embedding.compute_affinity(self.Z)

        if self.marginal_loss:
            constC, hC1, hC2 = init_matrix(
                self.PX, PZ, self.h0, self.T.sum(0), self.loss_fun, self.backend
            )
        else:
            q = self.T.sum(0)
            _, hC1, hC2, fC2t = init_matrix_semirelaxed(
                self.PX, PZ, self.h0, self.loss_fun, self.marginal_loss, self.backend
            )
            constC = q.expand([self.N, self.output_sam]) @ fC2t

        Loss = gwloss(constC, hC1, hC2, self.T, self.backend)
        return Loss

    def _update_T(self):
        """
        Computes the (fused) Gromov-Wasserstein optimal transport plan.
        """
        PZ = self.affinity_embedding.compute_affinity(self.Z)
        if self.alpha < 1:
            # If fused Gromov-Wasserstein is used (alpha<1), update centroids in input space to compute the linear OT term.
            centroids = barycenter_feat(self.X, self.T)
            M = torch.cdist(self.X, centroids, 2) ** 2
            if self.entropic_reg == 0.0:
                self.T = semirelaxed_fused_gromov_wasserstein(
                    M,
                    self.PX,
                    PZ,
                    self.h0,
                    loss_fun=self.loss_fun,
                    alpha=self.alpha,
                    G0=self.T,
                    marginal_loss=self.marginal_loss,
                    log=False,
                    verbose=False,
                ).detach()
            else:
                self.T = entropic_semirelaxed_fused_gromov_wasserstein(
                    M,
                    self.PX,
                    PZ,
                    self.h0,
                    loss_fun=self.loss_fun,
                    alpha=self.alpha,
                    epsilon=self.entropic_reg,
                    G0=self.T,
                    marginal_loss=self.marginal_loss,
                    stop_criterion="loss",
                    log=False,
                    verbose=False,
                ).detach()

        else:
            if self.entropic_reg == 0.0:

                self.T = semirelaxed_gromov_wasserstein(
                    self.PX,
                    PZ,
                    p=self.h0,
                    loss_fun=self.loss_fun,
                    G0=self.T,
                    marginal_loss=self.marginal_loss,
                    log=False,
                    verbose=False,
                ).detach()
            else:

                self.T = entropic_semirelaxed_gromov_wasserstein(
                    self.PX,
                    PZ,
                    self.h0,
                    loss_fun=self.loss_fun,
                    epsilon=self.entropic_reg,
                    G0=self.T,
                    marginal_loss=self.marginal_loss,
                    objective="exact",
                    stop_criterion="loss",
                    stop_timestep=1,
                    log=False,
                    verbose=False,
                )


class Clust_then_DR(AffinityBasedDataSummarizer):
    """
    This class solves sequentially clustering (using kmeans) and then dimensionality reduction.
    Given a list (or single) input dataset X, an integer n and a dimension d, it computes a n x d representation of the input.

    Parameters
    ----------
    affinity_data : "precomputed" or BaseAffinity
        The affinity in the input space (in X, corresponds to P_X).
        If affinity_data is "precomputed" then a affinity matrix (instead of a BaseAffinity object) is needed as input for the fit method.
    affinity_embedding : BaseAffinity
        The affinity in the embedding space (in Z, corresponds to Q_Z).
    output_sam : int, optional
        Number of rows of the embedding matrix, by default 10.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    loss_fun : {'kl_loss', 'kl_nomarg_loss', 'square_loss'}, optional
        Loss used in the DR problem, by default 'kl_nomarg_loss'.
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use, by default 'Adam'.
    lr : float, optional
        Learning rate for the (embedding) algorithm, by default 1.0.
    init : {'random'} or torch.Tensor of shape (output_sam, output_dim), optional
        Initialization for the embedding Z, default 'random'.
    verbose : bool, optional
        Verbosity, by default True.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-6.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    seed : int, optional
        Random seed, by default 0.
    device: str or torch.device type, optional
        Device to perform computations, by default 'cpu'.
    dtype: torch.dtype, optional
        dtype used throughout computations, by defaut torch.double.
    """

    def __init__(
        self,
        affinity_data=GramAffinity(centering=True),
        affinity_embedding=GramAffinity(),
        output_sam=10,
        output_dim=2,
        loss_fun="square_loss",
        optimizer="Adam",
        lr=1.0,
        init="random",
        verbose=True,
        tol=1e-6,
        max_iter=1000,
        seed=0,
        device="cpu",
        dtype=torch.double,
        init_T="spectral",
    ):

        super(Clust_then_DR, self).__init__(
            affinity_data=affinity_data,
            affinity_embedding=affinity_embedding,
            output_sam=output_sam,
            output_dim=output_dim,
            loss_fun=loss_fun,
            optimizer=optimizer,
            lr=lr,
            init=init,
            verbose=verbose,
            tol=tol,
            max_iter=max_iter,
            seed=seed,
            device=device,
            dtype=dtype,
            init_T=init_T,
        )

        if self.affinity_data == "precomputed":
            raise WrongParameter(
                "affinity_data cannot be precomputed in Clust_then_DR since it must be computed on clustered data points."
            )

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and returns the embedding.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_embedding="precomputed".
        y : None
            Ignored.

        Returns
        -------
        Z : torch.Tensor of shape (output_sam, output_dim)
            Embedding of the training data in low-dimensional space.
        """
        self._compute_affinity(X)

        # Clustering
        if isinstance(self.init_T, torch.Tensor):
            if self.init_T.shape[0] != X.shape[0] or self.init_T.shape[1] != self.output_sam:
                raise WrongParameter(
                    "init_T tensor must be of shape {0} but found {1}".format(
                        (self.N, self.output_sam), self.init_T.shape
                    )
                )
            self.T = self.init_T.to(dtype=self.dtype, device=self.device)
        if self.init_T == "spectral":
            clustering_model = SpectralClustering(
                n_clusters=self.output_sam,
                random_state=self.seed,
                n_init=10,
                affinity="precomputed",
            ).fit(self.PX.cpu())
            self.T = torch.eye(self.output_sam)[clustering_model.labels_].to(
                dtype=self.dtype, device=self.device
            )
        elif self.init_T == "kmeans":
            clustering_model = KMeans(
                n_clusters=self.output_sam, random_state=self.seed, n_init=10
            ).fit(X.cpu())
            self.T = torch.eye(self.output_sam)[clustering_model.labels_].to(
                dtype=self.dtype, device=self.device
            )
        else:
            raise WrongParameter('init_T must be either "kmeans" or precomputed as a torch tensor.')

        # Dimensionality Reduction
        self.X_reduced_sam = barycenter_feat(X, self.T)
        self._compute_affinity(self.X_reduced_sam)

        self._update_embedding()

        self.Z = self.Z.detach()
        return self.Z

    def _embed_loss(self):
        """
        Embedding loss function, use 'kl_nomarg_loss' for neighbor embedding methods and 'square_loss' for spectral methods.
        """
        if self.loss_fun == "kl_loss" or self.loss_fun == "kl_nomarg_loss":
            log_PZ = self.affinity_embedding.compute_log_affinity(self.Z)
            Loss = torch.nn.functional.kl_div(log_PZ, self.PX, reduction="sum")
            if self.loss_fun == "kl_loss":
                Loss += torch.logsumexp(log_PZ, dim=(0, 1))

        elif self.loss_fun == "binary_cross_entropy":
            PZ = self.affinity_embedding.compute_affinity(self.Z)
            Loss = torch.nn.functional.binary_cross_entropy(PZ, self.PX, reduction="sum")

        elif self.loss_fun == "square_loss":
            PZ = self.affinity_embedding.compute_affinity(self.Z)
            Loss = torch.pow(self.PX - PZ, 2).sum()
        return Loss


class DR_then_Clust:
    """
    This class solves sequentially dimensionality reduction and then clustering (using kmeans).
    Given a list (or single) input dataset X, an integer n and a dimension d, it computes a n x d representation of the input.

    Parameters
    ----------
    affinity_data : "precomputed" or BaseAffinity
        The affinity in the input space (in X, corresponds to P_X).
        If affinity_data is "precomputed" then a affinity matrix (instead of a BaseAffinity object) is needed as input for the fit method.
    affinity_embedding : BaseAffinity
        The affinity in the embedding space (in Z, corresponds to Q_Z).
    output_sam : int, optional
        Number of rows of the embedding matrix, by default 10.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    loss_fun : {'kl_loss', 'kl_nomarg_loss', 'square_loss'}, optional
        Loss used in the DR problem, by default 'square_loss'.
    optimizer : {'SGD', 'Adam', 'NAdam', 'RAdam'}, optional
        Which pytorch optimizer to use, by default 'Adam'.
    lr : float, optional
        Learning rate for the (embedding) algorithm, by default 1.0.
    init : {'random'} or torch.Tensor of shape (output_sam, output_dim), optional
        Initialization for the embedding Z, default 'random'.
    verbose : bool, optional
        Verbosity, by default True.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-6.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    seed : int, optional
        Random seed, by default 0.
    device: str or torch.device type, optional
        Device to perform computations, by default 'cpu'.
    dtype: torch.dtype, optional
        dtype used throughout computations, by defaut torch.double.
    """

    def __init__(
        self,
        affinity_data=GramAffinity(centering=True),
        affinity_embedding=GramAffinity(),
        output_sam=10,
        output_dim=2,
        loss_fun="square_loss",
        optimizer="Adam",
        lr=1.0,
        init="random",
        verbose=True,
        tol=1e-6,
        max_iter=1000,
        seed=0,
        device="cpu",
        dtype=torch.double,
        init_T="spectral",
    ):

        self.affinity_data = affinity_data
        self.affinity_embedding = affinity_embedding
        self.output_sam = output_sam
        self.output_dim = output_dim
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.lr = lr
        self.init = init
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.init_T = init_T

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and returns the embedding.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_embedding="precomputed".
        y : None
            Ignored.

        Returns
        -------
        Z : torch.Tensor of shape (output_sam, output_dim)
            Embedding of the training data in low-dimensional space.
        """

        # Dimensionality Reduction
        AM_model = AffinityMatcher(
            affinity_data=self.affinity_data,
            affinity_embedding=self.affinity_embedding,
            output_dim=self.output_dim,
            loss_fun=self.loss_fun,
            optimizer=self.optimizer,
            lr=self.lr,
            init=self.init,
            verbose=self.verbose,
            tol=self.tol,
            max_iter=self.max_iter,
            tolog=True,
        )

        Z_large_sam = AM_model.fit_transform(X)
        self.losses = AM_model.log_["loss"]
        PZ = AM_model.affinity_embedding.compute_affinity(Z_large_sam)

        # Clustering
        if self.init_T == "spectral":
            KZ = PZ
            if (
                KZ < 0
            ).any():  # Ensuring KX has positive coefficients for performing spectral clustering
                KZ -= KZ.min()

            # if self.device == 'cpu':
            clustering_model = SpectralClustering(
                n_clusters=self.output_sam,
                affinity="precomputed",
                random_state=self.seed,
                n_init=1,  # 10 before but super slow
            ).fit(KZ.cpu())

            """
            NOT AT ALL CONSISTENT WITH SCIPY EIGENDECOMPOSITION
            else: # we aim at doing spectral clustering as much as possible on GPU
                
                # compute normalize laplacian
                
                deg_ = KZ.sum(dim=1) ** (- 1. / 2.)
                I = torch.eye(KZ.shape[0], dtype=KZ.dtype, device=KZ.device)
                Lap_norm = I -  (KZ / torch.outer(deg_, deg_))
                # compute smallest k eigenvectors, dropping first
                
                eigval, eigvec = torch.lobpcg(Lap_norm, k=self.output_sam + 1, largest=False)
                spectral_embedding = eigvec * deg_[:, None]                
                clustering_model = KMeans(n_clusters=self.output_sam,
                        n_init=10,
                        random_state=self.seed,
                        verbose=True,
                    ).fit(spectral_embedding)
            """
        elif self.init_T == "kmeans":
            clustering_model = KMeans(
                n_clusters=self.output_sam, random_state=self.seed, n_init=10
            ).fit(Z_large_sam.cpu())
        else:
            raise WrongParameter('init_T must be either "spectral" or "kmeans".')

        self.T = torch.eye(self.output_sam)[clustering_model.labels_].to(
            dtype=self.dtype, device=self.device
        )
        self.Z = barycenter_feat(Z_large_sam, self.T)

        self.Z = self.Z.detach()
        return self.Z
