# -*- coding: utf-8 -*-
"""
Adaptation of the Semi-relaxed Gromov-Wasserstein utils from POT solvers.
"""

from ot.utils import list_to_array
from ot.backend import get_backend


def init_matrix_semirelaxed(C1, C2, p, loss_fun='square_loss', marginal_loss=True, nx=None):
    r"""Return loss matrices and tensors for semi-relaxed Gromov-Wasserstein fast computation

    Returns the value of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` with the
    selected loss function as the loss function of semi-relaxed Gromov-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`
    and adapted to the semi-relaxed problem where the second marginal is not a constant anymore.

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{T}`: A coupling between those two spaces

    The square-loss function :math:`L(a, b) = |a - b|^2` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a^2

                        f_2(b) &= b^2

                        h_1(a) &= a

                        h_2(b) &= 2b

    The kl-loss function :math:`L(a, b) = a \log\left(\frac{a}{b}\right) - a + b` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a \log(a) - a

                        f_2(b) &= b

                        h_1(a) &= a

                        h_2(b) &= \log(b)
    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6) adapted to srGW
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    fC2t: array-like, shape (nt, nt)
        :math:`\mathbf{f2}(\mathbf{C2})^\top` matrix in Eq. (6)


    .. _references-init-matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [48]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    if nx is None:
        C1, C2, p = list_to_array(C1, C2, p)
        nx = get_backend(C1, C2, p)

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            #return a * nx.log(a + 1e-15) - a
            return a * nx.log(a + 1e-100) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            #return nx.log(b + 1e-15)
            return nx.log(b + 1e-100)
            
    elif loss_fun == 'kl_nomarg_loss':
        def f1(a):
            #return a * nx.log(a + 1e-15)
            return a * nx.log(a + 1e-100)

        def f2(b):
            return nx.zeros(b.shape, type_as=b)

        def h1(a):
            return a

        def h2(b):
            #return nx.log(b + 1e-15)
            return nx.log(b + 1e-100)
    
    elif loss_fun == 'binary_cross_entropy':
        def f1(a):
            #return a * nx.log(a + 1e-15)
            return a * nx.log(a + 1e-100) + (1 - a) * nx.log(1 - a + 1e-100)

        def f2(b):
            return - nx.log(1 - b + 1e-100)

        def h1(a):
            return a

        def h2(b):
            #return nx.log(b + 1e-15)
            return nx.log((b / (1 - b)) + 1e-100)
    else:
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss', 'kl_no_marg'}.")

    if marginal_loss:
        constC = nx.dot(nx.dot(f1(C1), nx.reshape(p, (-1, 1))),
                        nx.ones((1, C2.shape[0]), type_as=p))
    else:
        constC = 0.
    hC1 = h1(C1)
    hC2 = h2(C2)
    fC2t = f2(C2).T
    return constC, hC1, hC2, fC2t