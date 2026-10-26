# -*- coding: utf-8 -*-
"""
Semi-relaxed Gromov-Wasserstein (CG and MD) solvers with low-rank structures.
"""

import numpy as np


from ot.utils import list_to_array, unif
from ot.optim import semirelaxed_cg, solve_1d_linesearch_quad
from ot.backend import get_backend


from ._utils import (
    init_matrix_semirelaxed_low_rank_structures,
    gwloss_low_rank_structures,
    gwggrad_low_rank_structures)

# %% conditional gradient solvers


def semirelaxed_gromov_wasserstein_low_rank_structures(
        C1, C2, p=None, loss_fun='square_loss', symmetric=None, log=False, G0=None,
        marginal_loss=False, max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the semi-relaxed Gromov-Wasserstein divergence transport from :math:`(\mathbf{C_1}, \mathbf{p})` to :math:`\mathbf{C_2}` (see [48]).

    If marginal loss = True, the function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_{\mathbf{T}} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    Else it solves the equivalent problem where constant marginal costs are removed (speed-up).

    Where :

    - :math:`\mathbf{C_1} = \mathbf{A_1} \mathbf{B_1}^\top`: Low-rank decomposition of the metric cost matrix in the source space
    - :math:`\mathbf{C_2}`= \mathbf{A_2} \mathbf{B_2}^\top: Low-rank decomposition of the metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space

    - `L`: loss function to account for the misfit between the similarity matrices

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    Parameters
    ----------
    C1 : array-like, either low-rank decomposition (2, ns, ds) or structure (ns, ns)
        Low-rank decomposition of the metric cost matrix in the source space
    C2 : array-like, either low-rank decomposition (2, nt, dt) or structure (nt, nt)
        Low-rank decomposition of the metric cost matrix in the target space
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

    """
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[1], type_as=C1)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    low_rank_C1 = (len(C1.shape) == 3)
    low_rank_C2 = (len(C2.shape) == 3)

    # force usage only if at least one of structures admit low-rank decomposition
    # otherwise one can simply use the vanilla srgw solver.

    #assert (low_rank_C1 or low_rank_C2)

    if symmetric is None:
        symmetric_C1 = nx.allclose(
            C1[0], C1[1], atol=1e-10) if low_rank_C1 else nx.allclose(C1, C1.T, atol=1e-10)
        symmetric_C2 = nx.allclose(
            C2[0], C2[1], atol=1e-10) if low_rank_C2 else nx.allclose(C2, C2.T, atol=1e-10)
        symmetric = symmetric_C1 and symmetric_C2

    if G0 is None:
        q = unif(C2.shape[1], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

    if marginal_loss:
        hC1, hC2, fC2t, C2_prod, constC, C1_prod = init_matrix_semirelaxed_low_rank_structures(
            C1, C2, p, low_rank_C1, low_rank_C2, loss_fun, marginal_loss, nx)

    else:
        hC1, hC2, fC2t, C2_prod = init_matrix_semirelaxed_low_rank_structures(
            C1, C2, p, low_rank_C1, low_rank_C2, loss_fun, marginal_loss, nx)
        constC = 0.

    low_rank_hC1 = low_rank_C1
    low_rank_hC2 = False if loss_fun == 'kl_loss' else low_rank_C2

    ones_p = nx.ones(p.shape[0], type_as=p)

    def f(G):
        qG = nx.sum(G, 0)
        if loss_fun in ['square_loss', 'binary_cross_entropy']:
            # 'square_loss' : fC2t does not preserve low-rank decomposition
            # same for 'binary_cross_entropy'
            marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        elif loss_fun == 'kl_loss':
            # 'kl_loss': preserve low.rank decomposition in fC2t but not in hC2
            if low_rank_C2:
                inner_marginal_product = nx.dot(nx.dot(qG, fC2t[0]), fC2t[1].T)
            else:
                inner_marginal_product = nx.dot(qG, fC2t)
            marginal_product = nx.outer(ones_p, inner_marginal_product)
        
        if marginal_loss:
            return gwloss_low_rank_structures(constC + marginal_product, hC1, hC2, G, low_rank_hC1, low_rank_hC2, nx)
        else:
            return gwloss_low_rank_structures(marginal_product, hC1, hC2, G, low_rank_hC1, low_rank_hC2, nx)
        
    if symmetric:
        def df(G):
            qG = nx.sum(G, 0)
            if loss_fun in ['square_loss', 'binary_cross_entropy']:
                marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
            elif loss_fun == 'kl_loss':
                if low_rank_C2:
                    inner_marginal_product = nx.dot(
                        nx.dot(qG, fC2t[0]), fC2t[1].T)
                else:
                    inner_marginal_product = nx.dot(qG, fC2t)
                marginal_product = nx.outer(ones_p, inner_marginal_product)
            
            if marginal_loss:
                return gwggrad_low_rank_structures(constC + marginal_product, hC1, hC2, G, low_rank_hC1, low_rank_hC2, nx)
            else:
                return gwggrad_low_rank_structures(marginal_product, hC1, hC2, G, low_rank_hC1, low_rank_hC2, nx)
    
    else:
        C1t = nx.flip(C1, axis=0) if low_rank_C1 else C1.T
        C2t = nx.flip(C2, axis=0) if low_rank_C2 else C2.T
        if marginal_loss:
            hC1t, hC2t, fC2, C2t_prod, constCt, C1t_prod = init_matrix_semirelaxed_low_rank_structures(
                C1t, C2t, p, low_rank_C1, low_rank_C2, loss_fun, marginal_loss, nx)
        else:
            hC1t, hC2t, fC2, C2t_prod = init_matrix_semirelaxed_low_rank_structures(
                C1t, C2t, p, low_rank_C1, low_rank_C2, loss_fun, marginal_loss, nx)

        def df(G):
            qG = nx.sum(G, 0)
            if loss_fun in ['square_loss', 'binary_cross_entropy']:
                marginal_product_1 = nx.outer(ones_p, nx.dot(qG, fC2t))
                marginal_product_2 = nx.outer(ones_p, nx.dot(qG, fC2))
            elif loss_fun == 'kl_loss':
                if low_rank_C2:
                    inner_marginal_product_1 = nx.dot(
                        nx.dot(qG, fC2t[0]), fC2t[1].T)
                    inner_marginal_product_2 = nx.dot(
                        nx.dot(qG, fC2[0]), fC2[1].T)
                else:
                    inner_marginal_product_1 = nx.dot(qG, fC2t)
                    inner_marginal_product_2 = nx.dot(qG, fC2)
                marginal_product_1 = nx.outer(ones_p, inner_marginal_product_1)
                marginal_product_2 = nx.outer(ones_p, inner_marginal_product_2)
            
            if marginal_loss:
                return 0.5 * (gwggrad_low_rank_structures(constC + marginal_product_1, hC1, hC2, G, low_rank_hC1, low_rank_hC2, nx) + gwggrad_low_rank_structures(constCt + marginal_product_2, hC1t, hC2t, G, low_rank_hC1, low_rank_hC2, nx))
            else:
                return 0.5 * (gwggrad_low_rank_structures(marginal_product_1, hC1, hC2, G, low_rank_hC1, low_rank_hC2, nx) + gwggrad_low_rank_structures(marginal_product_2, hC1t, hC2t, G, low_rank_hC1, low_rank_hC2, nx))

    def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
        return solve_semirelaxed_gromov_linesearch_low_rank_structures(
            G, deltaG, cost_G, hC1, hC2, ones_p, M=0., reg=1., fC2t=fC2t,
            low_rank1=low_rank_hC1, low_rank2=low_rank_hC2, nx=nx, **kwargs)

    if log:
        res, log = semirelaxed_cg(p, q, 0., 1., f, df, G0, line_search, log=True,
                                  numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)
        log['srgw_dist'] = log['loss'][-1]

        log['C2_prod'] = C2_prod
        log['C1_prod'] = C1_prod if marginal_loss else None

        return res, log
    else:
        return semirelaxed_cg(p, q, 0., 1., f, df, G0, line_search, log=False, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)




def solve_semirelaxed_gromov_linesearch_low_rank_structures(
        G, deltaG, cost_G, C1, C2, ones_p,
        M, reg, fC2t, low_rank1, low_rank2, alpha_min=None, alpha_max=None, nx=None, **kwargs):
    """
    Solve the linesearch in the Conditional Gradient iterations for the semi-relaxed Gromov-Wasserstein divergence.

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    cost_G : float
        Value of the cost at `G`
    C1 : array-like, either low-rank decomposition (2, ns, ds) or structure (ns, ns)
        Transformed Structure matrix in the source domain.
        Note that for the 'square_loss' and 'kl_loss', we provide hC1 from ot.gromov.init_matrix_semirelaxed
    C2 : array-like, either low-rank decomposition (2, nt, dt) or structure (nt, nt)
        Transformed Structure matrix in the source domain.
        Note that for the 'square_loss' and 'kl_loss', we provide hC2 from ot.gromov.init_matrix_semirelaxed
    ones_p: array-like (ns, 1)
        Array of ones of size ns
    M : array-like (ns,nt)
        Cost matrix between the features.
    reg : float
        Regularization parameter.
    fC2t: array-like, either low-rank decomposition (2, nt, dt) or structure (nt, nt)
        Transformed Structure matrix in the source domain.
        Note that for the 'square_loss' and 'kl_loss', we provide fC2t from ot.gromov.init_matrix_semirelaxed_low_rank_structures.
    low_rank1: bool, optional
        Either hC1 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    low_rank2: bool, optional
        Either hC2 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    
    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    cost_G : float
        The value of the cost for the next iteration

    """
    if nx is None:
        G, deltaG, C1, C2, M = list_to_array(G, deltaG, C1, C2, M)

        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, C1, C2)
        else:
            nx = get_backend(G, deltaG, C1, C2, M)

    if low_rank1 is None:
        low_rank1 = (len(C1.shape) == 3)

    if low_rank2 is None:
        low_rank2 = (len(C2.shape) == 3)

    low_rank_fC2 = (len(fC2t.shape) == 3)
    qG, qdeltaG = nx.sum(G, 0), nx.sum(deltaG, 0)
    # dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    # where actually C1=h(C1) and C2 = h(C2)
    # so available factorizations for 'square_loss' and 'kl_loss' differ.
    #
    # if loss_fun == 'square_loss':
    # compute `dot` using low-rank factorizations
    # and the product `nx.dot(nx.dot(hC1, G), hC2.T)`
    # where both hC1 and hC2 have low-rank factorizations
    #
    # if loss_fun == 'kl_loss:
    #
    # where only hC1 can benefit from a low-rank factorization
    # However in this case `fC2t` admits a low-rank factorization
    # so we may split computations in the following way:
    #   <A1.B1'.G. C2, G> = <B1'.G.C2; A1'.G>
    # and
    #   <1_n. q'. fC2t, G> = <1_n q' fC2t[0], G.fC2t[1]>
    #
    # For now I stick to the same implementation for 'square_loss' and 'kl_loss'
    # we need to go through the details of `a` and `b` to not have 2 times more terms

    if low_rank1 and low_rank2:
        if C1.shape[-1] < C2.shape[-1]:
            inner_dot_deltaG = nx.dot(nx.dot(C1[1].T, deltaG), C2[1])
            dot_deltaG = nx.dot(C1[0], nx.dot(inner_dot_deltaG, C2[0].T))

            inner_dot_G = nx.dot(nx.dot(C1[1].T, G), C2[1])
            dot_G = nx.dot(C1[0], nx.dot(inner_dot_G, C2[0].T))

        else:
            inner_dot_deltaG = nx.dot(C1[1].T, nx.dot(deltaG, C2[1]))
            dot_deltaG = nx.dot(nx.dot(C1[0], inner_dot_deltaG), C2[0].T)

            inner_dot_G = nx.dot(C1[1].T, nx.dot(G, C2[1]))
            dot_G = nx.dot(nx.dot(C1[0], inner_dot_G), C2[0].T)

    elif low_rank1:  # A1. B1'. (delta)G. C2.T
        inner_dot_deltaG = nx.dot(nx.dot(C1[1].T, deltaG), C2.T)
        dot_deltaG = nx.dot(C1[0], inner_dot_deltaG)

        inner_dot_G = nx.dot(nx.dot(C1[1].T, G), C2)
        dot_G = nx.dot(C1[0], inner_dot_G)

    elif low_rank2:  # C1. (delta)G. B2. A2'
        inner_dot_deltaG = nx.dot(C1, nx.dot(deltaG, C2[1]))
        dot_deltaG = nx.dot(inner_dot_deltaG, C2[0].T)

        inner_dot_G = nx.dot(C1, nx.dot(G, C2[1]))
        dot_G = nx.dot(inner_dot_G, C2[0].T)

    else:  # C1. (delta)G. C2.T
        dot_deltaG = nx.dot(nx.dot(C1, deltaG), C2.T)
        dot_G = nx.dot(nx.dot(C1, G), C2.T)

    if low_rank_fC2:
        inner_dot_qG = nx.dot(nx.dot(qG, fC2t[0]), fC2t[1].T)
        dot_qG = nx.outer(ones_p, inner_dot_qG)

        inner_dot_qdeltaG = nx.dot(nx.dot(qdeltaG, fC2t[0]), fC2t[1].T)
        dot_qdeltaG = nx.outer(ones_p, inner_dot_qdeltaG)

    else:
        dot_qG = nx.outer(ones_p, nx.dot(qG, fC2t))
        dot_qdeltaG = nx.outer(ones_p, nx.dot(qdeltaG, fC2t))

    a = reg * nx.sum((dot_qdeltaG - dot_deltaG) * deltaG)
    b = nx.sum(M * deltaG) + reg * (nx.sum((dot_qdeltaG - dot_deltaG)
                                           * G) + nx.sum((dot_qG - dot_G) * deltaG))

    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost can be deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha ** 2) + b * alpha

    return alpha, 1, cost_G


# %% entropic bregman projection based solvers

def entropic_semirelaxed_gromov_wasserstein_low_rank_structures(
        C1, C2, p=None, loss_fun='square_loss', epsilon=0.1, symmetric=None,
        G0=None, marginal_loss=True, objective='exact', max_iter=1e4, tol=1e-9,
        stop_criterion='plan', stop_timestep=1, log=False,
        verbose=False, **kwargs):
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

    - :math:`\mathbf{C_1} = \mathbf{A_1} \mathbf{B_1}^\top`: Low-rank decomposition of the metric cost matrix in the source space
    - :math:`\mathbf{C_2}`= \mathbf{A_2} \mathbf{B_2}^\top: Low-rank decomposition of the metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space

    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: negative entropy loss

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    Parameters
    ----------
    C1 : array-like, either low-rank decomposition (2, ns, ds) or structure (ns, ns)
        Low-rank decomposition of the metric cost matrix in the source space
    C2 : array-like, either low-rank decomposition (2, nt, dt) or structure (nt, nt)
        Low-rank decomposition of the metric cost matrix in the target space
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

    """
    assert objective in ['exact', 'entropic']

    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    low_rank_C1 = (len(C1.shape) == 3)
    low_rank_C2 = (len(C2.shape) == 3)

    # force usage only if at least one of structures admit low-rank decomposition
    # otherwise one can simply use the vanilla srgw solver.

    # assert (low_rank_C1 or low_rank_C2)

    if symmetric is None:
        symmetric_C1 = nx.allclose(
            C1[0], C1[1], atol=1e-10) if low_rank_C1 else nx.allclose(C1, C1.T, atol=1e-10)
        symmetric_C2 = nx.allclose(
            C2[0], C2[1], atol=1e-10) if low_rank_C2 else nx.allclose(C2, C2.T, atol=1e-10)
        symmetric = symmetric_C1 and symmetric_C2

    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

    if marginal_loss:
        hC1, hC2, fC2t, C2_prod, constC, C1_prod = init_matrix_semirelaxed_low_rank_structures(
            C1, C2, p, low_rank_C1, low_rank_C2, loss_fun, marginal_loss, nx)

    else:
        hC1, hC2, fC2t, C2_prod = init_matrix_semirelaxed_low_rank_structures(
            C1, C2, p, low_rank_C1, low_rank_C2, loss_fun, marginal_loss, nx)
        constC = 0.

    low_rank_hC1 = low_rank_C1
    low_rank_hC2 = False if loss_fun == 'kl_loss' else low_rank_C2

    ones_p = nx.ones(p.shape[0], type_as=p)

    if symmetric:
        def df(G):
            qG = nx.sum(G, 0)
            if loss_fun in ['square_loss', 'binary_cross_entropy']:
                marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
            elif loss_fun == 'kl_loss':
                if low_rank_C2:
                    inner_marginal_product = nx.dot(
                        nx.dot(qG, fC2t[0]), fC2t[1].T)
                else:
                    inner_marginal_product = nx.dot(qG, fC2t)
                marginal_product = nx.outer(ones_p, inner_marginal_product)
            return gwggrad_low_rank_structures(constC + marginal_product, hC1, hC2, G, low_rank_hC1, low_rank_hC2, nx)

    else:
        C1t = nx.flip(C1, axis=0) if low_rank_C1 else C1.T
        C2t = nx.flip(C2, axis=0) if low_rank_C2 else C2.T
        if marginal_loss:
            hC1t, hC2t, fC2, C2t_prod, constCt, C1t_prod = init_matrix_semirelaxed_low_rank_structures(
                C1t, C2t, p, low_rank_C1, low_rank_C2, loss_fun, marginal_loss, nx)
        else:
            hC1t, hC2t, fC2, C2t_prod = init_matrix_semirelaxed_low_rank_structures(
                C1t, C2t, p, low_rank_C1, low_rank_C2, loss_fun, marginal_loss, nx)

        def df(G):
            qG = nx.sum(G, 0)
            if loss_fun in ['square_loss', 'binary_cross_entropy']:
                marginal_product_1 = nx.outer(ones_p, nx.dot(qG, fC2t))
                marginal_product_2 = nx.outer(ones_p, nx.dot(qG, fC2))
            elif loss_fun == 'kl_loss':
                if low_rank_C2:
                    inner_marginal_product_1 = nx.dot(
                        nx.dot(qG, fC2t[0]), fC2t[1].T)
                    inner_marginal_product_2 = nx.dot(
                        nx.dot(qG, fC2[0]), fC2[1].T)
                else:
                    inner_marginal_product_1 = nx.dot(qG, fC2t)
                    inner_marginal_product_2 = nx.dot(qG, fC2)
                marginal_product_1 = nx.outer(ones_p, inner_marginal_product_1)
                marginal_product_2 = nx.outer(ones_p, inner_marginal_product_2)

            return 0.5 * (gwggrad_low_rank_structures(constC + marginal_product_1, hC1, hC2, G, low_rank_hC1, low_rank_hC2, nx) + gwggrad_low_rank_structures(constCt + marginal_product_2, hC1t, hC2t, G, low_rank_hC1, low_rank_hC2, nx))

    cpt = 0
    err = 1e15
    if stop_criterion == 'loss':
        loss_prev = 1e15

    G = G0

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Gprev = G
        # compute the kernel
        dG = df(G)
        if objective == 'exact':
            K = G * nx.exp(- dG / epsilon)
        else:
            K = nx.exp(- dG / epsilon)

        scaling = p / nx.sum(K, 1)
        G = nx.reshape(scaling, (-1, 1)) * K
        if cpt % stop_timestep == 0:
            if stop_criterion == 'plan':
                err = nx.norm(G - Gprev) / nx.norm(G)
            
            else: # access convergence w.r.t 'loss'
                if objective == 'exact':
                    loss = 0.5 * nx.sum(dG * G)
                else: # add entropic regularizer
                    dG_reg = (0.5 * dG) + epsilon * (nx.log(G) - 1.)
                    loss = nx.sum(dG_reg * G)
                err = nx.abs(loss - loss_prev) / nx.abs(loss)
                loss_prev = loss

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Rel. Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        # on purpose we do not output the regularized loss
        # if objective = 'entropic' to ease comparison with other srGW methods.

        qG = nx.sum(G, 0)
        if loss_fun in ['square_loss', 'binary_cross_entropy']:
            # 'square_loss' : fC2t does not preserve low-rank decomposition
            marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        elif loss_fun == 'kl_loss':
            # 'kl_loss': preserve low.rank decomposition in fC2t but not in hC2
            if low_rank_C2:
                inner_marginal_product = nx.dot(nx.dot(qG, fC2t[0]), fC2t[1].T)
            else:
                inner_marginal_product = nx.dot(qG, fC2t)
            marginal_product = nx.outer(ones_p, inner_marginal_product)

        log['srgw_dist'] = gwloss_low_rank_structures(
            constC + marginal_product, hC1, hC2, G, low_rank_hC1, low_rank_hC2, nx)
        return G, log
    else:
        return G