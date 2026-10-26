# -*- coding: utf-8 -*-
"""
Adaptation of the Semi-relaxed Gromov-Wasserstein from POT solvers.
Additional functionalities are the following:
    - support KL loss for both CG and MD solvers
    - allows to tackle the equivalent problem without the fixed marginal term for speed up
    - unifies both CG and MD solvers to use the objective function as stopping criterion.
"""

import numpy as np


from ot.utils import list_to_array, unif
from ot.optim import semirelaxed_cg
from ot.backend import get_backend

from ._utils import init_matrix_semirelaxed
from ot.gromov import gwloss, gwggrad, solve_semirelaxed_gromov_linesearch

# %% conditional gradient solvers


def semirelaxed_gromov_wasserstein(
    C1,
    C2,
    p=None,
    loss_fun="square_loss",
    symmetric=None,
    log=False,
    G0=None,
    marginal_loss=True,
    max_iter=1e4,
    tol_rel=1e-9,
    tol_abs=1e-9,
    **kwargs,
):
    r"""
    Returns the semi-relaxed Gromov-Wasserstein divergence transport from :math:`(\mathbf{C_1}, \mathbf{p})` to :math:`\mathbf{C_2}` (see [48]).

    If marginal_loss = True, the function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_{\mathbf{T}} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    Else it solves the equivalent problem where constant marginal costs are removed (speed-up).

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space

    - `L`: loss function to account for the misfit between the similarity matrices

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    marginal_loss: bool, optional
        Either to take into account the constant marginal terms or not.
    max_iter : int, optional
        Max number of iterations
    tol_rel : float, optional
        Stop threshold on relative error (>0)
    tol_abs : float, optional
        Stop threshold on absolute error (>0)
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Coupling between the two spaces that minimizes:

            :math:`\sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}`
    log : dict
        Convergence information and loss.

    References
    ----------
    .. [48]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

    # constC = 0. here if `marginal_loss=False`
    constC, hC1, hC2, fC2t = init_matrix_semirelaxed(C1, C2, p, loss_fun, marginal_loss, nx)

    ones_p = nx.ones(p.shape[0], type_as=p)

    def f(G):
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        if marginal_loss:
            return gwloss(constC + marginal_product, hC1, hC2, G, nx)
        else:
            return gwloss(marginal_product, hC1, hC2, G, nx)

    if symmetric:

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
            if marginal_loss:
                return gwggrad(constC + marginal_product, hC1, hC2, G, nx)
            else:
                return gwggrad(marginal_product, hC1, hC2, G, nx)

    else:
        constCt, hC1t, hC2t, fC2 = init_matrix_semirelaxed(
            C1.T, C2.T, p, loss_fun, marginal_loss, nx
        )

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product_1 = nx.outer(ones_p, nx.dot(qG, fC2t))
            marginal_product_2 = nx.outer(ones_p, nx.dot(qG, fC2))
            if marginal_loss:
                return 0.5 * (
                    gwggrad(constC + marginal_product_1, hC1, hC2, G, nx)
                    + gwggrad(constCt + marginal_product_2, hC1t, hC2t, G, nx)
                )
            else:
                return 0.5 * (
                    gwggrad(marginal_product_1, hC1, hC2, G, nx)
                    + gwggrad(marginal_product_2, hC1t, hC2t, G, nx)
                )

    def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
        return solve_semirelaxed_gromov_linesearch(
            G, deltaG, cost_G, hC1, hC2, ones_p, M=0.0, reg=1.0, fC2t=fC2t, nx=nx, **kwargs
        )

    if log:
        res, log = semirelaxed_cg(
            p,
            q,
            0.0,
            1.0,
            f,
            df,
            G0,
            line_search,
            log=True,
            numItermax=max_iter,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            **kwargs,
        )
        log["srgw_dist"] = log["loss"][-1]
        return res, log
    else:
        return semirelaxed_cg(
            p,
            q,
            0.0,
            1.0,
            f,
            df,
            G0,
            line_search,
            log=False,
            numItermax=max_iter,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            **kwargs,
        )


# %% entropic bregman projections based solvers


def entropic_semirelaxed_gromov_wasserstein(
    C1,
    C2,
    p=None,
    loss_fun="square_loss",
    epsilon=0.1,
    symmetric=None,
    G0=None,
    marginal_loss=True,
    objective="exact",
    max_iter=1e4,
    tol=1e-9,
    stop_criterion="plan",
    stop_timestep=1,
    log=False,
    verbose=False,
    **kwargs,
):
    r"""
    If objective='exact' returns the semi-relaxed gromov-wasserstein (srGW) divergence
    transport plan from :math:`(\mathbf{C_1}, \mathbf{p})` to :math:`\mathbf{C_2}`
    estimated using a Mirror Descent algorithm following the KL geometry.

    If marginal loss = True, the function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    Else it solves the equivalent problem where constant marginal costs are removed (speed-up).

    If objective='entropic' returns the entropic srGW transport plan estimated
    using a Mirror Descent algorithm following the KL geometry:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon H(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space

    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: negative entropy loss

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    verbose : bool, optional
        Print information along iterations
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    marginal_loss: bool, optional
        Either to take into account the constant marginal terms or not.
    objective: str, default is 'exact'.
        If set to 'exact', the solver solves for the exact srGW objective.
        Else if set to 'entropic', it solves for the entropic srGW objective.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error computed on the desired stopping criterion
    stop_criterion: str, optional. Default is 'plan'.
        Stopping criteration taking values in {'plan', 'loss'} to respectively
        access converge w.r.t relative variation of the plan norm or the loss.
    stop_timestep: int, optional. Default is 1.
        Evaluate stopping criterion at each iteration modulo `stop_timestep`.
        We can reduce the cost of convergence evaluation in O(ns * nt) by increasinng stop_timestep.
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
    Returns
    -------
    G : array-like, shape (`ns`, `nt`)
        Coupling between the two spaces that minimizes:

            :math:`\sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}`
    log : dict
        Convergence information and loss.

    References
    ----------
    .. [48]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    assert objective in ["exact", "entropic"]
    assert stop_criterion in ["plan", "loss"]
    assert isinstance(stop_timestep, int) and stop_timestep > 0

    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

    constC, hC1, hC2, fC2t = init_matrix_semirelaxed(C1, C2, p, loss_fun, marginal_loss, nx)

    ones_p = nx.ones(p.shape[0], type_as=p)

    if symmetric:

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
            if marginal_loss:
                return gwggrad(constC + marginal_product, hC1, hC2, G, nx)
            else:
                return gwggrad(marginal_product, hC1, hC2, G, nx)

    else:
        constCt, hC1t, hC2t, fC2 = init_matrix_semirelaxed(
            C1.T, C2.T, p, loss_fun, marginal_loss, nx
        )

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product_1 = nx.outer(ones_p, nx.dot(qG, fC2t))
            marginal_product_2 = nx.outer(ones_p, nx.dot(qG, fC2))
            if marginal_loss:
                return 0.5 * (
                    gwggrad(constC + marginal_product_1, hC1, hC2, G, nx)
                    + gwggrad(constCt + marginal_product_2, hC1t, hC2t, G, nx)
                )
            else:
                return 0.5 * (
                    gwggrad(marginal_product_1, hC1, hC2, G, nx)
                    + gwggrad(marginal_product_2, hC1t, hC2t, G, nx)
                )

    cpt = 0
    err = 1e15
    if stop_criterion == "loss":
        loss_prev = 1e15

    G = G0

    if log:
        log = {"err": []}

    while err > tol and cpt < max_iter:

        Gprev = G
        # compute the kernel
        dG = df(G)
        if objective == "exact":
            K = G * nx.exp(-dG / epsilon)
        else:
            K = nx.exp(-dG / epsilon)

        scaling = p / nx.sum(K, 1)
        G = nx.reshape(scaling, (-1, 1)) * K
        if cpt % stop_timestep == 0:

            if stop_criterion == "plan":
                err = nx.norm(G - Gprev) / nx.norm(G)

            else:  # access convergence w.r.t 'loss'
                if objective == "exact":
                    loss = 0.5 * nx.sum(dG * G)
                else:  # add entropic regularizer
                    dG_reg = (0.5 * dG) + epsilon * (nx.log(G) - 1.0)
                    loss = nx.sum(dG_reg * G)
                err = nx.abs(loss - loss_prev) / nx.abs(loss)
                loss_prev = loss

            if log:
                log["err"].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Rel. Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))

        cpt += 1

    if log:
        # on purpose we do not output the regularized loss
        # if objective = 'entropic' to ease comparison with other srGW methods.

        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        if marginal_loss:
            log["srgw_dist"] = gwloss(constC + marginal_product, hC1, hC2, G, nx)
        else:
            log["srgw_dist"] = gwloss(marginal_product, hC1, hC2, G, nx)

        return G, log
    else:
        return G
