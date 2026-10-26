import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from ot.gromov import gwloss, init_matrix
from local_ot.gromov._utils import init_matrix_semirelaxed

# from local_ot.fused_gromov._semirelaxed import (
#     semirelaxed_fused_gromov_wasserstein,
#     entropic_semirelaxed_fused_gromov_wasserstein,
# )
from ot.gromov import (
    entropic_fused_gromov_wasserstein,
    entropic_semirelaxed_fused_gromov_wasserstein,
    fused_gromov_wasserstein,
    semirelaxed_fused_gromov_wasserstein,
)

from src.affinities import GramAffinity

from ot.backend import get_backend

from src.utils import (
    KMeans,
)  # GPU friendly version of kmeans used for spectral clustering
from src.clust_dr import AffinityBasedDataSummarizer
from src.hsic_loss import hsic
from src.kernels import KernelFactory

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


class SDR(AffinityBasedDataSummarizer):
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
    hsic_kernel_z : str
        Kernel used for computing similarities in the mebdding space for HSIC computation
    hsic_kernel_y : str
        Kernel used for computing similarities in the output space for HSIC computation
    fgw_loss: str
        Loss used for the superivision term in the FGW objective. This can be "square_loss" for a squared loss on continuous outputs or
        "ce_loss" for a cross-entropy loss on class labels
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
    accept_reject_T: bool, optional
        If True, a candidate transport plan produced by the T-subproblem solver is only accepted when it
        strictly decreases the objective at fixed Z; otherwise the previous plan is kept and the update is
        counted as a rejection. This makes the T-step monotone even when the (entropic / semi-relaxed)
        solver overshoots. By default True.
    t_gate : {"J", "F"}, optional
        Which objective the accept/reject test uses, by default "J".

        * "J" -- the complete SDR objective ``J(T) = F(T) - lambda_hsic * CKA(Z, Y(T))``. Since the
          CKA term depends on T when ``output_sam < n_samples``, this is the only setting under
          which the T-step is monotone in the objective that defines the method.
        * "F" -- only the solver's own (fused) GW objective. Never vetoes a valid solver step, but
          exercises no control over the supervision term.

        Ignored when ``objective="fgw"`` (there is no CKA term, so J == F).
    learn_L: bool, optional
        If True and `output_sam` matches the number of input samples, learns the projection map
        used by `transform` for out-of-sample embedding. If False, skips learning L even in the
        pure DR case.

    [2] Titouan Vayer, Laetitia Chapel, Rémi Flamary, Romain Tavenard and Nicolas Courty
    "Optimal Transport for structured data with application on graphs" International Conference on Machine Learning (ICML). 2019.

    """

    def __init__(
        self,
        alpha,
        affinity_data=GramAffinity(centering=True),
        affinity_embedding=GramAffinity(),
        hsic_kernel_z="rbf",
        hsic_kernel_y="rbf",
        use_rff=False,
        n_features=256,
        fgw_loss="square_loss",
        output_sam=10,
        output_dim=2,
        loss_fun="square_loss",
        objective="sdr",
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
        max_iter_outer=10,
        marginal_loss=False,
        accept_reject_T=False,
        t_gate="J",
        entropic_reg=0.0,
        early_stopping=True,
        lambda_hsic=0.5,
        beta=0.5,
        learn_L=True,
    ):

        super(SDR, self).__init__(
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
        self.accept_reject_T = accept_reject_T
        if t_gate not in ("J", "F"):
            raise WrongParameter('t_gate must be either "J" or "F".')
        self.t_gate = t_gate
        self.entropic_reg = entropic_reg
        self.running_loss = torch.inf
        self.running_Z = None
        self.early_stopping = early_stopping
        self.lambda_hsic = lambda_hsic
        self.beta = beta
        self.learn_L = learn_L
        self.objective = objective

        self.inner_loss_history = []
        self.outer_loss_history = []
        self.last_loss_terms = None
        self._current_outer_iter = None

        # Accept/reject bookkeeping for the T-subproblem
        self.T_update_history = []
        self.n_T_updates = 0
        self.n_T_rejections = 0

        if self.affinity_data == "precomputed":
            if self.alpha < 1:
                raise WrongParameter(
                    'When affinity_data="precomputed" the fused GW alpha parameter must be set to 1 (no linear term).'
                )

        self.hsic_kernel_z = hsic_kernel_z
        self.hsic_kernel_y = hsic_kernel_y
        self.fgw_loss = fgw_loss
        self.use_rff = use_rff
        self.n_features = n_features

        if self.hsic_kernel_z not in ["rbf", "rq"] and use_rff:
            raise ValueError("RFF only supported for stationary kernels (rbf and rq)")
        if self.hsic_kernel_y not in ["rbf", "rq"] and use_rff:
            raise ValueError("RFF only supported for stationary kernels (rbf and rq)")

    def fit_transform(self, X, y):
        """Fit X into an embedded space and returns the embedding.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_embedding="precomputed".
        y : None
            torch.Tensor of shape (n_samples, n_output_feat) or a torch.Tensor of shape (n_samples,) containing 1D outputs.

        Returns
        -------
        Z : torch.Tensor of shape (output_sam, output_dim)
            Embedding of the training data in low-dimensional space.
        """
        self.backend = get_backend(X)
        self.X = X
        self.y = y
        self.losses = []
        self.inner_loss_history = []
        self.outer_loss_history = []
        self.last_loss_terms = None
        self._current_outer_iter = None
        self.running_loss = torch.inf
        self.running_Z = None
        self.T_update_history = []
        self.n_T_updates = 0
        self.n_T_rejections = 0

        self._compute_affinity(X)

        self._init_T()

        # Learn the projection map only in the pure DR case when requested.
        if self.learn_L and self.output_sam == X.shape[-2]:
            self._init_L()
        else:
            self.L = None

        # Outer loop for BCD
        for t in range(self.max_iter_outer):
            self._current_outer_iter = t

            if self.verbose:
                print(f"--- {t}-th outer loop ---")

            self._update_embedding()

            with torch.no_grad():
                self._update_T()

            # To learn mappings along with the embedding and coupling matrix
            if self.L is not None:
                with torch.no_grad():
                    self._update_L()

            with torch.no_grad():
                outer_loss_terms = self._compute_loss_terms()
            self._record_loss_terms(self.outer_loss_history, outer_loss_terms, outer_iter=t)
            embed_loss = outer_loss_terms["objective"]
            if (embed_loss > self.running_loss) and self.early_stopping:
                self.Z = self.running_Z
                if self.verbose:
                    print(f"--- Breaking : BCD iteration increasing the loss ---")
                break
            else:
                self.running_loss = embed_loss
                self.running_Z = self.Z

        self.Z = self.Z.detach()

        if self.verbose and self.accept_reject_T:
            reasons = {}
            for h in self.T_update_history:
                if not h["accepted"]:
                    reasons[h["reason"]] = reasons.get(h["reason"], 0) + 1
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
            print(
                f"--- T accept/reject : {self.n_T_rejections}/{self.n_T_updates} "
                f"candidate updates rejected{' (' + breakdown + ')' if breakdown else ''} ---"
            )

        # In the pure DR case where a projection map L is learned, return the
        # RKHS-representable embedding Z = K L (built from the same L that
        # transform() uses) instead of the raw embedding. This guarantees the
        # returned training embedding equals transform(X_train), so train and
        # test embeddings share one space -- essential for out-of-sample
        # prediction, especially with entropic regularization where the raw Z
        # does not admit a consistent kernel-map extension.
        if self.L is not None:
            with torch.no_grad():
                Z_repr = self.Phi @ self.L if self.use_rff else self.K @ self.L
                self.Z = Z_repr.detach()

        return self.Z, self.L

    def _record_loss_terms(self, history, loss_terms, outer_iter=None, inner_iter=None):
        entry = {
            "objective": float(loss_terms["objective"].detach().item()),
            "structural_loss": float(loss_terms["structural_loss"].detach().item()),
            "supervision_loss": float(loss_terms["supervision_loss"].detach().item()),
        }
        if outer_iter is not None:
            entry["outer_iter"] = outer_iter
        if inner_iter is not None:
            entry["inner_iter"] = inner_iter
        history.append(entry)
        self.last_loss_terms = entry

    def _update_embedding(self):
        """
        Optimize the embedding while logging the full loss decomposition at each inner step.
        """
        self.Z.requires_grad = True
        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for i in pbar:
            self.optimizer.zero_grad()
            loss_terms = self._compute_loss_terms()
            loss = loss_terms["objective"]
            if torch.isnan(loss):
                raise ValueError("NaN in embedding loss")
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())
            self._record_loss_terms(
                self.inner_loss_history,
                loss_terms,
                outer_iter=self._current_outer_iter,
                inner_iter=i,
            )

            if i > 1:
                delta = abs(self.losses[-1] - self.losses[-2]) / abs(self.losses[-2])
                if delta < self.tol:
                    if self.verbose:
                        print("---------- delta loss convergence ----------")
                    break
                if self.verbose:
                    pbar.set_description(
                        "Loss : "
                        f"{float(self.losses[-1]): .3e}, "
                        f"struct : {self.last_loss_terms['structural_loss']: .3e}, "
                        f"sup : {self.last_loss_terms['supervision_loss']: .3e}, "
                        f"delta : {float(delta): .3e} "
                    )

    def transform(self, X_star):
        """
        X_star: (b, ds)
        Returns Z_star: (b, dt)
        """
        if self.L is None:
            raise RuntimeError(
                "Projection map L is unavailable. Fit SDR with learn_L=True in the pure DR case "
                "to enable out-of-sample transforms."
            )

        if self.use_rff:
            phi_x_star = self.rff.phi(X_star)
            Z_star = phi_x_star @ self.L
        else:
            kernel = KernelFactory("rbf")
            K_star, _ = kernel.forward(X_star, self.X, **{"sigma": self.sigma})  # (b, ns)
            Z_star = K_star.to(dtype=self.dtype, device=self.device) @ self.L  # (b, dt)

        return Z_star

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

    def _cka_value(self, T=None):
        """
        CKA/HSIC supervision term C = CKA(Z, Y(T)) at the current Z.

        `T` defaults to `self.T`; passing an explicit plan lets the accept/reject gate evaluate
        the supervision term at a *candidate* coupling. Note that when `output_sam < n_samples`
        the labels are pushed through the plan, so C genuinely depends on T -- which is exactly
        why gating on the structural term alone does not control the complete objective.

        Gradients are not detached: `_compute_loss_terms` relies on this for the Z-step.
        """
        # NOTE: sigma means two different things in the two cases
        # 1) For RFF case, sigma -> Standard deviation of the Gaussian from which frequncies are sampled
        # 2) Non-RFF case, sigma -> Lengthscale of the RBF kernel
        if self.use_rff:
            hsic_kwargs = {"sigma": 1.0}
        else:
            hsic_kwargs = {
                "sigma": None,
                "alpha": 1.0,
            }  # Alpha weighting of large-scale and small-scale mixtures in the RQ kernels

        if self.output_sam == self.X.shape[-2]:
            # Balanced case: labels are not pushed through the plan, so C is independent of T.
            return hsic(
                self.Z,
                self.y,
                kernel_z=self.hsic_kernel_z,
                kernel_y=self.hsic_kernel_y,
                device=self.device,
                dtype=self.dtype,
                **hsic_kwargs,
            )
        elif self.output_sam < self.X.shape[-2]:
            return hsic(
                self.Z,
                self.y,
                T=self.T if T is None else T,
                kernel_z=self.hsic_kernel_z,
                kernel_y=self.hsic_kernel_y,
                device=self.device,
                dtype=self.dtype,
                **hsic_kwargs,
            )
        else:
            raise ValueError("output_sam cannot exceed the number of input samples.")

    def _compute_loss_terms(self):
        """
        Compute the structural, supervision, and total objective terms.
        """
        PZ = self.affinity_embedding.compute_affinity(self.Z)

        if self.PX.shape[-2] != PZ.shape[-2]:
            constC, hC1, hC2, _ = init_matrix_semirelaxed(
                self.PX, PZ, self.h0, self.loss_fun, self.marginal_loss, self.backend
            )
            # This is a scalar in the semi-relaxed case, so we convert it to a tensor for donwstream computations
            constC = torch.tensor(constC, dtype=self.dtype, device=self.device)
        else:
            constC, hC1, hC2 = init_matrix(
                self.PX, PZ, self.h0, self.T.sum(0), self.loss_fun, self.backend
            )

        # Compute the GW loss
        constC, hC1, hC2 = (
            constC.to(dtype=self.dtype, device=self.device),
            hC1.to(dtype=self.dtype, device=self.device),
            hC2.to(dtype=self.dtype, device=self.device),
        )
        structural_loss = gwloss(constC, hC1, hC2, self.T, self.backend)

        if self.objective == "sdr":
            # Use the HSIC loss for the supervision term instead of the standard feature cost
            supervision_loss = self._cka_value()

            total_loss = (self.alpha * structural_loss) - (self.lambda_hsic * supervision_loss)

        # Compute the standard FGW version of the supervision loss
        elif self.objective == "fgw":
            if self.fgw_loss == "square_loss":
                M = self._square_loss()
            elif self.fgw_loss == "ce_loss":
                M = self._ce_loss()

            # Normalize the coupling matrix
            total_mass = self.T.sum().clamp_min(1e-12)

            # Compute the linear supervision term
            supervision_loss = (M * self.T).sum() / total_mass

            total_loss = (self.alpha * structural_loss) + (1 - self.alpha) * supervision_loss

        return {
            "objective": total_loss,
            "structural_loss": structural_loss,
            "supervision_loss": supervision_loss,
        }

    def _embed_loss(self):
        """
        Gromov-Wasserstein loss function.
        """
        return self._compute_loss_terms()["objective"]

    def _init_L(self, lambda_init=1e-2, jitter=1e-8):
        """
        Initialize L by kernel ridge regression: (K + lambda I) L = Z
        Shapes:
        Non-RFF case-> X: (ns, ds), Z: (nt, dt), L: (ns, dt)
        RFF case-> X: (ns, ds), Z: (nt, dt), L: (D, dt)
        """

        def _stable_cholesky(K_mat, base_reg, base_jitter):
            n = K_mat.shape[0]
            I = torch.eye(n, device=K_mat.device, dtype=K_mat.dtype)
            reg = float(base_reg + base_jitter)
            last_error = None
            for _ in range(8):
                A = K_mat + reg * I
                try:
                    chol = torch.linalg.cholesky(A)
                    return chol, reg
                except RuntimeError as err:
                    last_error = err
                    reg *= 10.0
            raise RuntimeError(
                f"Failed to build a positive-definite kernel system even after increasing "
                f"regularization up to {reg:.1e}."
            ) from last_error

        # Use Random Fourier Features for kernel computation over X
        if self.use_rff:
            # Compute kernel matrix K on all training points
            kernel = KernelFactory("rbf")
            K, self.rff = kernel.forward_with_rff(self.X, m=self.n_features)
            K = (0.5 * (K + K.T)).to(dtype=self.dtype, device=self.device)  # ensure symmetry
            self.Phi = self.rff.phi(self.X)
            self.A_cholesky, self.lambda_L = _stable_cholesky(K, lambda_init, jitter)
            B = self.Phi.T @ self.Z.detach()
            self.L = torch.cholesky_solve(B, self.A_cholesky)
        else:
            ns = self.X.shape[0]

            # Compute kernel matrix K on all training points
            kernel = KernelFactory("rbf")
            K, self.sigma = kernel.forward(self.X, **{"sigma": None})
            self.K = (0.5 * (K + K.T)).to(dtype=self.dtype, device=self.device)  # ensure symmetry
            self.A_cholesky, self.lambda_L = _stable_cholesky(self.K, lambda_init, jitter)
            self.L = torch.cholesky_solve(self.Z.detach(), self.A_cholesky)

    def _update_L(self):
        """
        Fix 2: proximal / projection step to keep Z close to the RKHS map class.

        Solves: L* = argmin_L ||K L - Z||^2 + lam_L ||L||^2
        Then:   Z_proj = K L*
                Z <- (1-beta) Z + beta Z_proj    (soft projection)
        """
        Z = self.Z

        if self.use_rff:
            # Best-response L* via Cholesky solve
            B = self.Phi.T @ self.Z.detach()
            L_star = torch.cholesky_solve(B, self.A_cholesky)  # (D, dt)

            # Project Z onto the RKHS class: Z_proj = K L*
            Z_proj = self.Phi @ L_star  # (ns, dt)
        else:
            # Best-response L* via Cholesky solve
            L_star = torch.cholesky_solve(Z.detach(), self.A_cholesky)  # (ns, dt)

            # Project Z onto the RKHS class: Z_proj = K L*
            Z_proj = self.K @ L_star  # (ns, dt)

        # Soft projection of Z (in-place, no grads)
        with torch.no_grad():
            Z.mul_(1.0 - self.beta).add_(self.beta * Z_proj)

        # Keep L synced (store detached)
        self.L = L_star.detach()

    def _complete_gw_value(self, PZ, T, marginal_loss=True):
        """
        Complete (semi-relaxed) GW value of a plan `T` at fixed embedding affinity `PZ`.

        `gwloss(constC, hC1, hC2, T)` evaluates only ``<constC - hC1 T hC2^T, T>`` and therefore
        omits the term ``q^T f2(PZ) q`` with ``q = T.sum(0)``. In the *semi-relaxed* problem ``q``
        is a free variable, so that term is not constant and changes the minimizer -- it must be
        included. This mirrors exactly how the reference srGW solver evaluates its own objective,
        ``gwloss(constC + outer(1_p, q @ fC2t), hC1, hC2, T)``; note that `marginal_loss` only
        toggles the ``p``-dependent constant ``constC``, never the ``q``-dependent term.

        Verified to match the brute-force ``sum_ijkl L(PX[i,k], PZ[j,l]) T[i,j] T[k,l]`` to
        machine precision.
        """
        constC, hC1, hC2, fC2t = init_matrix_semirelaxed(
            self.PX, PZ, self.h0, self.loss_fun, marginal_loss, self.backend
        )
        constC = torch.as_tensor(constC, dtype=self.dtype, device=self.device)
        hC1 = hC1.to(dtype=self.dtype, device=self.device)
        hC2 = hC2.to(dtype=self.dtype, device=self.device)
        fC2t = fC2t.to(dtype=self.dtype, device=self.device)

        q = T.sum(0)
        marginal_product = torch.outer(torch.ones_like(self.h0), q @ fC2t)
        return gwloss(constC + marginal_product, hC1, hC2, T, self.backend)

    def _solver_objective(self, T, PZ, M):
        """
        Value of ``F(T)``, the objective that the T-subproblem solver actually minimizes.

        This is deliberately *not* `_compute_loss_terms()["objective"]`: that quantity drops the
        ``q``-dependent marginal term (see `_complete_gw_value`) and normalizes the linear term by
        the transport mass, so it is a different function of ``T`` than the one `_solve_T`
        descends. `M` is passed in and held fixed across the before/after evaluation because the
        solver treats it as a constant cost matrix.

        Subclasses that override `_solve_T` should override this to mirror their own solver.
        """
        if self.PX.shape[-2] != PZ.shape[-2]:
            # Semi-relaxed solvers (POT's srFGW) always include the full marginal terms.
            structural = self._complete_gw_value(PZ, T, marginal_loss=True)
        else:
            # Balanced case: both marginals are fixed, so `init_matrix` already yields the
            # complete objective and `gwloss` needs no correction.
            constC, hC1, hC2 = init_matrix(
                self.PX, PZ, self.h0, T.sum(0), self.loss_fun, self.backend
            )
            constC = torch.as_tensor(constC, dtype=self.dtype, device=self.device)
            hC1 = hC1.to(dtype=self.dtype, device=self.device)
            hC2 = hC2.to(dtype=self.dtype, device=self.device)
            structural = gwloss(constC, hC1, hC2, T, self.backend)

        if M is None:
            return structural
        return self.alpha * structural + (1.0 - self.alpha) * (M * T).sum()

    def _t_gate_objective(self, T, PZ, M):
        """
        Value of the objective the accept/reject gate tests, as selected by `t_gate`.

        ``t_gate="J"`` (default) tests the *complete* SDR objective

            J(T) = F(T) - lambda_hsic * C(T),   C(T) = CKA(Z, Y(T))

        Because ``C`` depends on ``T`` whenever ``output_sam < n_samples``, a step that decreases
        ``F`` can still increase ``J``; gating on ``J`` rules that out, so the T-step is monotone
        in the objective that actually defines the method rather than only in the solver's
        surrogate. The solver is not required to optimize ``J`` -- the gate merely *evaluates* it
        at the incumbent and candidate plans, which is cheap at fixed Z.

        ``t_gate="F"`` tests only the solver's own objective. That never vetoes a valid solver
        step, but it provides no control over the supervision term.

        For ``objective="fgw"`` there is no CKA term in the model, so J == F and the two settings
        coincide.

        Caveat, deliberately not hidden: the Z-step descends
        `_compute_loss_terms()["objective"]`, which uses the marginal-free structural term and
        omits the linear label cost. So a J-gate makes the *T-step* monotone in J, but the T-step
        and Z-step still descend slightly different functionals; a single monotone descent
        guarantee for the whole BCD loop additionally requires reconciling those two.
        """
        value = self._solver_objective(T, PZ, M)

        if self.t_gate == "J" and self.objective == "sdr":
            value = value - self.lambda_hsic * self._cka_value(T=T)

        return value

    def _update_T(self):
        """
        Update the transport plan with an accept/reject (monotone descent) safeguard.

        The T-subproblem solver is a descent step only in exact arithmetic; the entropic and
        semi-relaxed solvers can return a plan that *increases* the objective at fixed Z (bad
        initialization, epsilon too large, line-search failure). Here the FGW objective is
        evaluated before and after the candidate update -- Z is held fixed, so the two values are
        directly comparable -- and the candidate is kept only if it strictly decreases it.

        Rejections are counted in `self.n_T_rejections` and each update is logged in
        `self.T_update_history`. The gate uses `_t_gate_objective` (the objective the solver
        minimizes); the embedding objective from `_compute_loss_terms` is logged alongside it for
        reference but is *not* what the decision is based on.
        """
        T_prev = self.T
        T_cand, PZ, M = self._solve_T(return_context=True)
        self.n_T_updates += 1

        with torch.no_grad():
            fgw_prev = float(self._t_gate_objective(T_prev, PZ, M).detach().item())
            fgw_cand = float(self._t_gate_objective(T_cand, PZ, M).detach().item())
            delta_T = float((T_cand - T_prev).abs().sum().item())

            # Record the solver's own objective F alongside the gated value so that the
            # F-vs-J disagreement is visible even when the gate runs on J.
            solver_prev = float(self._solver_objective(T_prev, PZ, M).detach().item())
            solver_cand = float(self._solver_objective(T_cand, PZ, M).detach().item())

        # A NaN/inf candidate is never acceptable.
        finite = bool(torch.isfinite(torch.as_tensor(fgw_cand)))
        accepted = finite and (fgw_cand < fgw_prev)

        if not finite:
            reason = "non_finite"
        elif accepted:
            reason = "accepted"
        elif delta_T <= 1e-10 * max(1.0, float(T_prev.abs().sum().item())):
            # The solver returned (numerically) the same plan -- it has converged or stalled,
            # rather than proposing an uphill step. Rejecting this is a no-op.
            reason = "no_progress"
        else:
            reason = "no_decrease"

        if accepted or not self.accept_reject_T:
            self.T = T_cand
            fgw_after = fgw_cand
        else:
            self.T = T_prev
            self.n_T_rejections += 1
            fgw_after = fgw_prev
            if self.verbose:
                print(
                    f"--- T update rejected [{reason}] (FGW {fgw_prev: .6e} -> {fgw_cand: .6e}), "
                    f"keeping previous plan "
                    f"[{self.n_T_rejections}/{self.n_T_updates} rejected] ---"
                )

        with torch.no_grad():
            embed_terms = self._compute_loss_terms()

        self.T_update_history.append(
            {
                "outer_iter": self._current_outer_iter,
                "gate": self.t_gate,
                # Gated objective (J by default, F when t_gate="F").
                "fgw_before": fgw_prev,
                "fgw_candidate": fgw_cand,
                "fgw_after": fgw_after,
                # Solver's own objective F, for comparison against the gated value.
                "solver_F_before": solver_prev,
                "solver_F_candidate": solver_cand,
                # True when the candidate decreased F but was still blocked -- i.e. the
                # supervision term is what vetoed it.
                "F_decreased_but_rejected": bool(
                    (solver_cand < solver_prev) and not accepted and self.accept_reject_T
                ),
                "accepted": accepted,
                # "accepted" | "no_decrease" (uphill step, genuinely blocked)
                # | "no_progress" (solver returned the same plan) | "non_finite"
                "reason": reason,
                "delta_T": delta_T,
                # False when a non-decreasing candidate was kept anyway because the
                # safeguard is switched off.
                "enforced": self.accept_reject_T,
                "embedding_objective_after": float(embed_terms["objective"].detach().item()),
            }
        )

    def _solve_T(self, return_context=False):
        """
        Computes the (fused) Gromov-Wasserstein optimal transport plan.

        Parameters
        ----------
        return_context : bool, optional
            If True, also return the quantities the solver held fixed (`PZ` and the linear cost
            `M`) so that `_update_T` can score the candidate against the very same objective the
            solver minimized. By default False.

        Returns
        -------
        T : torch.Tensor of shape (n_samples, output_sam)
            The candidate transport plan. Assignment to `self.T` is handled by `_update_T`,
            which applies the accept/reject safeguard.
        PZ, M : torch.Tensor
            Only when ``return_context=True``.
        """
        PZ = self.affinity_embedding.compute_affinity(self.Z).detach()
        if self.alpha < 1:
            # If fused Gromov-Wasserstein is used (alpha<1), update centroids in input space to compute the linear OT term.
            if self.fgw_loss == "square_loss":
                M = self._square_loss()
            elif self.fgw_loss == "ce_loss":
                M = self._ce_loss()
            else:
                raise ValueError(
                    "Incompatible loss function. Choose between 'square_loss' and 'ce_loss'"
                )

            if self.PX.shape[-2] != PZ.shape[-2]:
                if self.entropic_reg == 0.0:
                    T_cand = semirelaxed_fused_gromov_wasserstein(
                        M,
                        self.PX,
                        PZ,
                        self.h0,
                        loss_fun=self.loss_fun,
                        alpha=self.alpha,
                        G0=self.T,
                        log=False,
                    ).detach()
                else:
                    T_cand = entropic_semirelaxed_fused_gromov_wasserstein(
                        M,
                        self.PX,
                        PZ,
                        self.h0,
                        loss_fun=self.loss_fun,
                        alpha=self.alpha,
                        epsilon=self.entropic_reg,
                        G0=self.T,
                        log=False,
                    ).detach()
            else:
                if self.entropic_reg == 0.0:
                    T_cand = fused_gromov_wasserstein(
                        M,
                        self.PX,
                        PZ,
                        self.h0,
                        self.T.sum(0),
                        loss_fun=self.loss_fun,
                        alpha=self.alpha,
                        G0=self.T,
                        log=False,
                    ).detach()
                else:
                    T_cand = entropic_fused_gromov_wasserstein(
                        M,
                        self.PX,
                        PZ,
                        self.h0,
                        self.T.sum(0),
                        loss_fun=self.loss_fun,
                        alpha=self.alpha,
                        epsilon=self.entropic_reg,
                        G0=self.T,
                        log=False,
                    ).detach()

            M = M.detach()
        else:
            raise ValueError(
                "Alpha has to be 0 < alpha < 1 for Fused Gromov Wasserstein objective."
            )

        return (T_cand, PZ, M) if return_context else T_cand

    def _square_loss(self):
        # Compute prototype targets g* from current coupling T
        T = self.T  # (n, m)
        y = self.y  # (n,) or (n, d)

        # column masses (m,)
        col_mass = T.sum(dim=0).clamp_min(1e-12)

        # g*: (m,) if y is (n,), else (m, d)
        if y.dim() == 1:
            g = (T.t() @ y) / col_mass
        else:
            g = (T.t() @ y) / col_mass.unsqueeze(-1)

        # Build the linear OT cost matrix M using squared L2 loss ---
        # M: (n, m)
        if y.dim() == 1:
            M = (y.unsqueeze(1) - g.unsqueeze(0)).pow(2)
        else:
            M = torch.cdist(y, g, p=2).pow(2)

        M = M / M.median().clamp_min(1e-12)  # normalize cost matrix
        return M

    def _ce_loss(self):
        # K = int(self.y.max() + 1)
        # Y_onehot = torch.eye(K)[self.y]  # (n, K)
        y_idx = self.y.to(device=self.device, dtype=torch.long)
        K = int(torch.max(y_idx).item() + 1)
        Y_onehot = torch.eye(K, dtype=self.dtype, device=self.device)[y_idx]  # (n, K)

        # Barycentric class-prototype distributions in target space
        col_mass = self.T.sum(dim=0).clamp_min(1e-12)  # (1, m)
        G = (self.T.T @ Y_onehot) / col_mass[:, None]  # (m, K)
        G = torch.clip(G, 1e-12, 1.0)
        G = G / G.sum(axis=1, keepdims=True)  # ensure distribution

        # Pairwise cross-entropy cost matrix M[i,j] = -log G[j, y[i]]
        M = -torch.log(G[:, y_idx].T)
        return M.to(dtype=self.dtype, device=self.device)
