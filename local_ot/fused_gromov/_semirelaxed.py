from ot.gromov._semirelaxed import solve_semirelaxed_gromov_linesearch
from ot.gromov._utils import init_matrix_semirelaxed, gwloss, gwggrad, semirelaxed_init_plan
from ot.optim import semirelaxed_cg
from ot.utils import list_to_array, unif, get_backend


def semirelaxed_fused_gromov_wasserstein(
    M,
    C1,
    C2,
    p=None,
    loss_fun="square_loss",
    symmetric=None,
    alpha=0.5,
    G0=None,
    log=False,
    max_iter=1e4,
    tol_rel=1e-9,
    tol_abs=1e-9,
    random_state=0,
    # **kwargs,
):
    r"""
    Computes the semi-relaxed Fused Gromov-Wasserstein transport between two graphs (see [48]).

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) T_{i,j} T_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{p}` source weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices


    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like of shape (ns,nt) or string, optional
        If `G0=None` the initial transport plan of the solver is :math:`\mathbf{p} \frac{\mathbf{1}_{nt}}{nt}^\top`.
        If G0 is a tensor it must satisfy marginal constraints and will be
        used as initial transport of the solver.
        if G0 is a string it will be interpreted as a method for
        :func:`ot.gromov.semirelaxed_init_plan` taking values in "product",
        "random_product", "random", "fluid", "fluid_soft", "spectral",
        "spectral_soft", "kmeans", "kmeans_soft".
    log : bool, optional
        record log if True
    max_iter : int, optional
        Max number of iterations
    tol_rel : float, optional
        Stop threshold on relative error (>0)
    tol_abs : float, optional
        Stop threshold on absolute error (>0)
    random_state: int, optional
        Random seed used in stochastic initialization methods.
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    gamma : array-like, shape (`ns`, `nt`)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.


    .. _references-semirelaxed-fused-gromov-wasserstein:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
        "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
        International Conference on Learning Representations (ICLR), 2022.

    .. [62] H. Van Assel, C. Vincent-Cuaz, T. Vayer, R. Flamary, N. Courty.
        "Interpolating between Clustering and Dimensionality Reduction with
        Gromov-Wasserstein". NeurIPS 2023 Workshop OTML.
    """
    arr = [M, C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)

    if (G0 is not None) and (not isinstance(G0, str)):
        arr.append(G0)

    nx = get_backend(*arr)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)

    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    elif isinstance(G0, str):
        G0 = semirelaxed_init_plan(C1, C2, p, M, alpha, G0, random_state=random_state, nx=nx)
        q = nx.sum(G0, 0)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

    constC, hC1, hC2, fC2t = init_matrix_semirelaxed(C1, C2, p, loss_fun, nx)

    ones_p = nx.ones(p.shape[0], type_as=p)

    def f(G):
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        return gwloss(constC + marginal_product, hC1, hC2, G, nx)

    if symmetric:

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
            return gwggrad(constC + marginal_product, hC1, hC2, G, nx)

    else:
        constCt, hC1t, hC2t, fC2 = init_matrix_semirelaxed(C1.T, C2.T, p, loss_fun, nx)

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product_1 = nx.outer(ones_p, nx.dot(qG, fC2t))
            marginal_product_2 = nx.outer(ones_p, nx.dot(qG, fC2))
            return 0.5 * (
                gwggrad(constC + marginal_product_1, hC1, hC2, G, nx)
                + gwggrad(constCt + marginal_product_2, hC1t, hC2t, G, nx)
            )

    def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
        return solve_semirelaxed_gromov_linesearch(
            G,
            deltaG,
            cost_G,
            hC1,
            hC2,
            ones_p,
            M=(1 - alpha) * M,
            reg=alpha,
            fC2t=fC2t,
            nx=nx,
            **kwargs,
        )

    if log:
        res, log = semirelaxed_cg(
            p,
            q,
            (1 - alpha) * M,
            alpha,
            f,
            df,
            G0,
            line_search,
            log=True,
            numItermax=max_iter,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            # **kwargs,
        )
        log["srfgw_dist"] = log["loss"][-1]
        return res, log
    else:
        return semirelaxed_cg(
            p,
            q,
            (1 - alpha) * M,
            alpha,
            f,
            df,
            G0,
            line_search,
            log=False,
            numItermax=max_iter,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            # **kwargs,
        )


def entropic_semirelaxed_fused_gromov_wasserstein(
    M,
    C1,
    C2,
    p=None,
    loss_fun="square_loss",
    symmetric=None,
    epsilon=0.1,
    alpha=0.5,
    G0=None,
    max_iter=1e4,
    tol=1e-9,
    log=False,
    verbose=False,
    random_state=0,
):
    r"""
    Computes the entropic-regularized semi-relaxed FGW transport between two graphs (see :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`)
    estimated using a Mirror Descent algorithm following the KL geometry.

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix between features
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}` source weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices


    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space
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
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like of shape (ns,nt) or string, optional
        If `G0=None` the initial transport plan of the solver is :math:`\mathbf{p} \frac{\mathbf{1}_{nt}}{nt}^\top`.
        If G0 is a tensor it must satisfy marginal constraints and will be
        used as initial transport of the solver.
        if G0 is a string it will be interpreted as a method for
        :func:`ot.gromov.semirelaxed_init_plan` taking values in "product",
        "random_product", "random", "fluid", "fluid_soft", "spectral",
        "spectral_soft", "kmeans", "kmeans_soft".
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error computed on transport plans
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
    random_state: int, optional
        Random seed used in stochastic initialization methods.

    Returns
    -------
    G : array-like, shape (`ns`, `nt`)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.


    .. _references-semirelaxed-fused-gromov-wasserstein:
    References
    ----------
    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
        "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
        International Conference on Learning Representations (ICLR), 2022.
    """
    arr = [M, C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)

    if (G0 is not None) and (not isinstance(G0, str)):
        arr.append(G0)

    nx = get_backend(*arr)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)

    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    elif isinstance(G0, str):
        G0 = semirelaxed_init_plan(C1, C2, p, M, alpha, G0, random_state=random_state, nx=nx)
        q = nx.sum(G0, 0)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

    constC, hC1, hC2, fC2t = init_matrix_semirelaxed(C1, C2, p, loss_fun, nx)

    ones_p = nx.ones(p.shape[0], type_as=p)
    dM = (1 - alpha) * M
    if symmetric:

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
            return alpha * gwggrad(constC + marginal_product, hC1, hC2, G, nx) + dM

    else:
        constCt, hC1t, hC2t, fC2 = init_matrix_semirelaxed(C1.T, C2.T, p, loss_fun, nx)

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product_1 = nx.outer(ones_p, nx.dot(qG, fC2t))
            marginal_product_2 = nx.outer(ones_p, nx.dot(qG, fC2))
            return (
                0.5
                * alpha
                * (
                    gwggrad(constC + marginal_product_1, hC1, hC2, G, nx)
                    + gwggrad(constCt + marginal_product_2, hC1t, hC2t, G, nx)
                )
                + dM
            )

    cpt = 0
    err = 1e15
    G = G0

    if log:
        log = {"err": []}

    while err > tol and cpt < max_iter:
        Gprev = G
        # compute the kernel
        K = G * nx.exp(-df(G) / epsilon)
        scaling = p / nx.sum(K, 1)
        G = nx.reshape(scaling, (-1, 1)) * K
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(G - Gprev)

            if log:
                log["err"].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))

        cpt += 1

    if log:
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        log["lin_loss"] = nx.sum(M * G) * (1 - alpha)
        log["quad_loss"] = alpha * gwloss(constC + marginal_product, hC1, hC2, G, nx)
        log["srfgw_dist"] = log["lin_loss"] + log["quad_loss"]
        return G, log
    else:
        return G
