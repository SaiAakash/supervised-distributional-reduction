import numpy as np
from sklearn.manifold import MDS
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from sklearn.manifold import MDS
import torch.nn.functional as F
import random


def init_bounds(f, n: int, begin=None, end=None, device="cpu"):
    if begin is None:
        begin = torch.ones(n, dtype=torch.double, device=device)
    else:
        begin = begin * torch.ones(n, dtype=torch.double, device=device)

    if end is None:
        end = torch.ones(n, dtype=torch.double, device=device)
    else:
        end = end * torch.ones(n, dtype=torch.double, device=device)

    out_begin = f(begin) > 0
    while out_begin.any():
        end[out_begin] = torch.min(end[out_begin], begin[out_begin])
        begin[out_begin] /= 2
        out_begin = f(begin) > 0

    out_end = f(end) < 0
    while out_end.any():
        begin[out_end] = torch.max(begin[out_end], end[out_end])
        end[out_end] *= 2
        out_end = f(end) < 0

    return begin, end


def false_position(
    f,
    n: int,
    begin: torch.Tensor = None,
    end: torch.Tensor = None,
    max_iter: int = 1000,
    tol: float = 1e-9,
    verbose: bool = False,
    device="cpu",
):
    """
    Performs the false position method to find the root of an increasing function f.

    Parameters
    ----------
    f: function
        function which root should be computed
    n: int
        size of the input of f
    begin: tensor, shape (n), optional
        initial lower bound of the root
    begin: tensor, shape (n), optional
        initial upper bound of the root
    max_iter: int
        maximum iterations of search
    tol: float
        precision threshold at which the algorithm stops
    verbose: bool
        if True, prints the mean of current bounds
    """

    begin, end = init_bounds(f=f, n=n, begin=begin, end=end, device=device)
    f_begin, f_end = f(begin), f(end)
    m = begin - ((begin - end) / (f(begin) - f(end))) * f(begin)
    fm = f(m)

    pbar = tqdm(range(max_iter), disable=not verbose)
    for _ in pbar:
        if torch.max(torch.abs(fm)) < tol:
            break
        sam = fm * f_begin > 0
        begin = sam * m + (~sam) * begin
        f_begin = sam * fm + (~sam) * f_begin
        end = (~sam) * m + sam * end
        f_end = (~sam) * fm + sam * f_end
        m = begin - ((begin - end) / (f_begin - f_end)) * f_begin
        fm = f(m)

        if verbose:
            mean_f = fm.mean().item()
            std_f = fm.std().item()
            pbar.set_description(
                f"f values : {float(mean_f): .3e} ({float(std_f): .3e}), "
                f"begin mean : {float(begin.mean().item()): .6e}, "
                f"end mean : {float(end.mean().item()): .6e} "
            )
    return m, begin, end


def log_selfsink(
    C: torch.Tensor,
    eps: float = 1.0,
    f: torch.Tensor = None,
    tol: float = 1e-5,
    max_iter: int = 1000,
    student: bool = False,
    tolog: bool = False,
):
    """
    Performs Sinkhorn iterations in log domain to solve the entropic "self" (or "symmetric") OT problem with symmetric cost C and entropic regularization epsilon.
    Returns the transport plan and dual variable at convergence.

    Parameters
    ----------
    C: array (n,n)
        symmetric distance matrix
    eps: float
        entropic regularization coefficient
    f: tensor, shape (n), optional
        initial dual variable
    tol: float, optional
        precision threshold at which the algorithm stops
    max_iter: int, optional
        maximum number of Sinkhorn iterations
    student: bool, optional
        if True, a Student-t kernel is considered instead of Gaussian
    tolog: bool
        if True, log and returns intermediate variables
    """
    n = C.shape[0]

    # Allows a warm-start if a dual variable f is provided
    f = torch.zeros(n) if f is None else f.clone().detach()

    if tolog:
        log = {}
        log["f"] = [f.clone().detach()]

    # If student is True, considers the Student-t kernel instead of Gaussian
    if student:
        C = torch.log(1 + C)

    # Sinkhorn iterations
    for k in range(max_iter + 1):
        f = 0.5 * (f - eps * torch.logsumexp((f - C) / eps, -1))

        if tolog:
            log["f"].append(f.clone().detach())

        if torch.isnan(f).any():
            raise Exception(f"NaN in self-Sinkhorn dual variable at iteration {k}")

        log_T = (f[:, None] + f[None, :] - C) / eps
        if (torch.abs(torch.exp(torch.logsumexp(log_T, -1)) - 1) < tol).all():
            break

        if k == max_iter - 1:
            print("---------- Max iter attained ----------")

    if tolog:
        return (f[:, None] + f[None, :] - C) / eps, f, log
    else:
        return (f[:, None] + f[None, :] - C) / eps, f


def entropy(P: torch.Tensor, log: bool = False, ax: int = -1):
    """
    Returns the entropy of P along axis ax, supports log domain input.

    Parameters
    ----------
    P: array (n,n)
        input data
    log: bool
        if True, assumes that P is in log domain
    ax: int
        axis on which entropy is computed
    """
    if log:
        return -(torch.exp(P) * (P - 1)).sum(ax)
    else:
        return -(P * (torch.log(P) - 1)).sum(ax)


def kl_div(P: torch.Tensor, K: torch.Tensor, log: bool = False):
    """
    Returns the Kullback-Leibler divergence between P and K, supports log domain input for both matrices.

    Parameters
    ----------
    P: array
        input data
    K: array
        input data
    log: bool
        if True, assumes that P and K are in log domain
    """
    if log:
        return (torch.exp(P) * (P - K - 1)).sum()
    else:
        return (P * (torch.log(P / K) - 1)).sum()


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(nn.Module):
    # PCA implementation in torch that matches the scikit-learn implementation
    # see https://github.com/gngdb/pytorch-pca
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_  # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


def barycenter_feat(X, T):
    h_ = T.sum(0)
    zeros = torch.argwhere(h_ == 0)[:, 0]
    h_[zeros] = 1.0
    Xbar = T.T @ X
    Xbar /= h_[:, None]
    return Xbar


def barycenter_structure(C, T, reduce_size=False):
    h_ = T.sum(0)
    prod = T.T @ C @ T
    if reduce_size:
        non_zeros = torch.argwhere(h_ != 0)[:, 0]
        h = h_[non_zeros]
        Cbar = prod[non_zeros, :][:, non_zeros]
        Cbar /= h[:, None] * h[None, :]
    else:
        zeros = torch.argwhere(h_ == 0)[:, 0]
        h_[zeros] = 1.0
        Cbar = prod / (h_[:, None] * h_[None, :])

    return Cbar


def barycenter_graph(C, T):
    h_ = T.sum(0)
    sc = torch.diag(1 / h_)
    C_ = sc @ T.T @ C @ T @ sc
    return C_


def plot_gwdr(Z, centroid, T, Y, zoom=0.2, thres=0.005, title="", ax=None):
    p = centroid.shape[-1]
    n = centroid.shape[0]
    c, s = plan_color(T, Y)
    ids = torch.where(T.sum(0) > thres)[0]
    ax.scatter(
        Z[ids, 0],
        Z[ids, 1],
        cmap=plt.cm.get_cmap("tab10"),
        alpha=0.5,
        c=c[ids],
        s=s[ids],
    )
    Xbar_im = centroid.reshape(n, int(p ** (0.5)), int(p ** (0.5)))
    for _, i in enumerate(ids):
        img = Image.fromarray(Xbar_im[i].numpy())
        ab = AnnotationBbox(
            OffsetImage(img, zoom=1.5e-1 * np.sqrt(s[i]) * zoom, cmap="gray"),
            (Z[i, 0], Z[i, 1]),
            frameon=True,
        )
        ax.add_artist(ab)
    title = f"{title}"
    ax.set_title(title)


def plot_graph(C, ax=None, binary=False, s=None, c=None):
    # C = np.array((C + C.T) / 2)
    x = MDS(dissimilarity="precomputed", random_state=0).fit_transform(1 - C)
    if ax is None:
        ax = plt
    for j in range(C.shape[0]):
        for i in range(j):
            if binary:
                if C[i, j] > 0:
                    ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=0.2, color="k")
            else:  # connection intensity proportional to C[i,j]
                ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=C[i, j], color="k")
    n = x.shape[0]
    # if c is None:
    #     c = np.arange(n)/n
    ax.scatter(x[:, 0], x[:, 1], c=c, s=s, zorder=10, edgecolors="k", cmap="rainbow")
    # for i,point in enumerate(x):
    #     ax.annotate(i, (point[0], point[1]), c='red')


def plan_color(T, Y):
    encoded_Y = F.one_hot(Y)
    weighted_encoded_Y = T.T @ encoded_Y.to(dtype=T.dtype)
    labels = torch.argmax(weighted_encoded_Y, dim=1)
    s = T.sum(0)
    return labels, s


# %%
class KMeans(object):
    def __init__(
        self,
        n_clusters=8,
        n_init=10,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=0,
        verbose=True,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.verbose = verbose
        self.labels_ = None
        self.cluster_centers_ = None

    def pairwise_dists(self, X, Y):

        return torch.cdist(X, Y, p=2)

    def init_centroids(self, X):
        if self.verbose:
            print("-- init centroids --")
        random.seed(self.random_state)

        if self.init == "random":
            centroids = X[random.sample(range(X.shape[0]), self.n_clusters), :]

        elif self.init == "k-means++":

            centroids = torch.zeros((self.n_clusters, X.shape[-1]), dtype=X.dtype, device=X.device)
            indices = set(range(X.shape[0]))

            for i in range(self.n_clusters):

                if i == 0:
                    idx = random.sample(range(X.shape[0]), self.n_clusters)[0]
                    centroids[i] = X[idx].clone()
                    indices.remove(idx)
                else:

                    l_indices = list(indices)
                    if i == 1:  # only one centroid still
                        distances = self.pairwise_dists(X[l_indices], centroids[0][None, :])
                    else:
                        distances = self.pairwise_dists(X[l_indices], centroids[:i])

                    idx = distances.sum(dim=1).argmax().item()
                    centroids[i] = X[l_indices[idx]]
                    indices.remove(l_indices[idx])

        return centroids

    def fit(self, X):
        n_samples = X.shape[0]

        self.inertia = None

        for run_it in range(self.n_init):
            centroids = self.init_centroids(X)

            for it in tqdm(range(self.max_iter), desc="fit kmeans (seed = %s)" % run_it):
                distances = self.pairwise_dists(centroids, X)

                labels = distances.argmin(dim=1)

                new_centroids = torch.zeros(
                    (self.n_clusters, X.shape[-1]), dtype=X.dtype, device=X.device
                )
                for i in range(self.n_clusters):
                    indices = torch.where(labels == i)[0]

                    if len(indices) > 0:
                        new_centroids[i, :] = X[indices].mean(dim=0)

                    else:  # handle empty cluster

                        new_centroids[i, :] = X[random.sample(range(n_samples), 1), :]

                diff_distances = 0.0
                for i in range(self.n_clusters):
                    diff_distances += self.pairwise_dists(
                        centroids[i, None], new_centroids[i, None]
                    ).item()

                if self.verbose:
                    print(f"Seed : {run_it} / step: {it} / diff_distances: {diff_distances}")
                centroids = new_centroids.clone()
                if diff_distances < self.tol:
                    break

            distances = self.pairwise_dists(centroids, X)

            labels = distances.argmin(dim=0)

            inertia = 0.0
            for i in range(self.n_clusters):
                indices = torch.where(labels == i)[0]
                inertia += distances[i, indices].sum().item()

            if (self.inertia == None) or (inertia < self.inertia):
                self.inertia = inertia
                self.labels_ = labels.clone()
                self.cluster_centers_ = centroids.clone()

            if self.verbose:
                print("Iteration: {} - Best Inertia: {}".format(run_it, self.inertia))

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def predict(self, X):
        distances = self.transform(X)
        return distances.argmin(dim=1)

    def transform(self, X):
        distances = self.pairwise_dists(self.cluster_centers_, X)

        return distances
