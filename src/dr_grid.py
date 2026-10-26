"""
Illustration of srGW projections on regular grids
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from local_ot.gromov._semirelaxed import (
    semirelaxed_gromov_wasserstein,
    entropic_semirelaxed_gromov_wasserstein,
)

from local_ot.gromov_low_rank._semirelaxed import (
    semirelaxed_gromov_wasserstein_low_rank_structures,
    entropic_semirelaxed_gromov_wasserstein_low_rank_structures,
)

from src.affinities import GramAffinity


def grid(vmax=1, n_points=1000):
    ar = np.linspace(-vmax, vmax, n_points)
    xv, yv = np.meshgrid(ar, ar)
    return np.array([xv.flatten(), yv.flatten()]).T


def DR_grid(
    X,
    Kx,
    affinity_data,
    affinity_embedding,
    h=None,
    vmax=1,
    pixels_per_row=100,
    loss_fun="square_loss",
    epsilon=1e0,
    max_iter=1000,
    tol=1e-5,
    init="random",
    verbose=False,
    objective="entropic",
    marginal_loss=False,
    log=False,
    return_grid=False,
    power=1.0,
):
    # Default marginal
    device = X.device
    if h is None:
        h = torch.ones(X.shape[0], dtype=X.dtype, device=device)

    # Input similarity matrix
    if Kx is None:
        low_rank_Kx = isinstance(affinity_embedding, GramAffinity)
        if low_rank_Kx:
            X = X - X.mean(0)
            Kx = torch.stack([X, X], 0)
        else:
            Kx = affinity_data.compute_affinity(X)
    else:
        low_rank_Kx = len(Kx.shape) == 3

    # Grid similarity matrix
    Z = torch.Tensor(grid(vmax=vmax, n_points=pixels_per_row)).to(
        dtype=X.dtype, device=device
    )

    low_rank_Kz = isinstance(affinity_data, GramAffinity)

    if low_rank_Kz:
        Kz = torch.stack([Z, Z], 0)
    else:
        Kz = affinity_embedding.compute_affinity(Z)

    # Init OT plan
    if init == "random":
        G0 = (
            torch.rand(X.shape[0], pixels_per_row**2, dtype=X.dtype, device=device)
            ** power
        )
        G0 /= G0.sum(-1, keepdim=True)
        G0 *= h[:, None]
    elif isinstance(init, torch.Tensor):
        G0 = init

    def srGW_operator(CX, CZ, h):
        if low_rank_Kx or low_rank_Kz:
            if epsilon == 0.0:
                return semirelaxed_gromov_wasserstein_low_rank_structures(
                    CX,
                    CZ,
                    h,
                    loss_fun=loss_fun,
                    max_iter=max_iter,
                    tol_rel=tol,
                    G0=G0,
                    verbose=verbose,
                    marginal_loss=marginal_loss,
                    log=log,
                )
            else:
                return entropic_semirelaxed_gromov_wasserstein_low_rank_structures(
                    CX,
                    CZ,
                    h,
                    loss_fun=loss_fun,
                    epsilon=epsilon,
                    max_iter=max_iter,
                    tol=tol,
                    G0=G0,
                    verbose=verbose,
                    objective=objective,
                    stop_criterion="loss",
                    stop_timestep=1,
                    marginal_loss=marginal_loss,
                    log=log,
                )

        else:
            if epsilon == 0.0:
                return semirelaxed_gromov_wasserstein(
                    CX,
                    CZ,
                    h,
                    loss_fun=loss_fun,
                    max_iter=max_iter,
                    tol_rel=tol,
                    G0=G0,
                    verbose=verbose,
                    marginal_loss=marginal_loss,
                    log=log,
                )
            else:
                return entropic_semirelaxed_gromov_wasserstein(
                    CX,
                    CZ,
                    h,
                    loss_fun=loss_fun,
                    epsilon=epsilon,
                    max_iter=max_iter,
                    tol=tol,
                    G0=G0,
                    verbose=verbose,
                    objective=objective,
                    marginal_loss=marginal_loss,
                    stop_criterion="loss",
                    stop_timestep=1,
                    log=log,
                )

    if return_grid:
        return srGW_operator(Kx, Kz, h), Z, Kz
    else:
        return srGW_operator(Kx, Kz, h)


def plot_grid(T, Y, cmap="tab10"):
    marg0 = T.sum(0)
    alpha = marg0 / marg0.max()
    cm = plt.colormaps[cmap]
    Y = Y.to(dtype=torch.float)
    Y /= Y.max()
    rgbsamples = cm(Y)
    gridcolors = (T / marg0).T @ rgbsamples
    gridcolors[:, 3] = alpha
    n_pixels = int(np.sqrt(gridcolors.shape[0]))
    return gridcolors.reshape(n_pixels, n_pixels, 4)


def plot_grid2(T, Y, cmap="tab10"):

    hgrid = int(np.sqrt(T.shape[1]))

    marg0 = T.sum(0)
    alpha = marg0 / marg0.max()
    cm = plt.colormaps[cmap]
    rgbsamples = cm(Y)
    gridcolors = (T / marg0).T @ rgbsamples
    gridcolors[:, 3] = alpha

    plt.imshow(gridcolors.reshape(hgrid, hgrid, 4))
    plt.axis("off")
    plt.axis("equal")