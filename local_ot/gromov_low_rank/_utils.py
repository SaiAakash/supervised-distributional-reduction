# -*- coding: utf-8 -*-

"""
Gromov-Wasserstein utils with low-rank structures
"""

from ot.utils import list_to_array
from ot.backend import get_backend


def tensor_product_low_rank_structures(
        constC, hC1, hC2, T, low_rank1=None, low_rank2=None, nx=None):
    r"""Return the tensor for Gromov-Wasserstein fast computation


    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, if low_rank1 expects (2, ns, ds) else expects (ns, ns)
        low-rank decomposition of :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, if low_rank2 expects (2, nt, dt) else expects (nt, nt)
        low-rank decomposition of :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    low_rank1: bool, optional
        Either hC1 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    low_rank2: bool, optional
        Either hC2 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    tens : array-like, shape (`ns`, `nt`)
        :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` tensor-matrix multiplication result

   
    """
    if nx is None:
        constC, hC1, hC2, T = list_to_array(constC, hC1, hC2, T)
        nx = get_backend(constC, hC1, hC2, T)

    if low_rank1 is None:
        low_rank1 = (len(hC1.shape) == 3)

    if low_rank2 is None:
        low_rank2 = (len(hC2.shape) == 3)

    # compute the low-rank product h(C1).T.h(C2)^t when possible

    if low_rank1 and low_rank2:

        if hC1.shape[-1] < hC2.shape[-1]:
            inner = nx.dot(nx.dot(hC1[1].T, T), hC2[1])
            outer = nx.dot(hC1[0], nx.dot(inner, hC2[0].T))

        else:
            inner = nx.dot(hC1[1].T, nx.dot(T, hC2[1]))
            outer = nx.dot(nx.dot(hC1[0], inner), hC2[0].T)

    elif low_rank1:
        inner = nx.dot(hC1[1].T, T)
        outer = nx.dot(hC1[0], nx.dot(inner, hC2))

    elif low_rank2:
        inner = nx.dot(T, hC2[0])
        outer = nx.dot(nx.dot(hC1, inner), hC2[1].T)

    else:
        outer = nx.dot(nx.dot(hC1, T), hC2.T)

    return constC - outer


def gwloss_low_rank_structures(
        constC, hC1, hC2, T, low_rank1=None, low_rank2=None, nx=None):
    r"""Return the Loss for Gromov-Wasserstein

    
    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, if low_rank1 expects (2, ns, ds) else expects (ns, ns)
        low-rank decomposition of :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, if low_rank2 expects (2, nt, dt) else expects (nt, nt)
        low-rank decomposition of :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`
    low_rank1: bool, optional
        Either hC1 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    low_rank2: bool, optional
        Either hC2 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    loss : float
        Gromov-Wasserstein loss

    
    """
    if nx is None:
        constC, hC1, hC2, T = list_to_array(constC, hC1, hC2, T)

    if low_rank1 is None:
        low_rank1 = (len(hC1.shape) == 3)

    if low_rank2 is None:
        low_rank2 = (len(hC2.shape) == 3)

    # operations assume that n * m operations imply the biggest computational burden
    # compute the crossed term <h(C1) . T . h(C2), T>
    if low_rank1 and low_rank2:
        # low-rank factorizations of Scetbon & al, 2023.

        if hC1.shape[-1] < hC2.shape[-1]:
            tens1 = nx.dot(nx.dot(hC1[0].T, T), hC2[1])
            tens2 = nx.dot(nx.dot(hC1[1].T, T), hC2[0])

        else:
            tens1 = nx.dot(hC1[0].T, nx.dot(T, hC2[1]))
            tens2 = nx.dot(hC1[1].T, nx.dot(T, hC2[0]))

    elif low_rank1:
        # <A1. B1' . T . hC2, T > = <B1'. T. hC2, A1' . T>
        tens1 = nx.dot(nx.dot(hC1[1].T, T), hC2)
        tens2 = nx.dot(hC1[0].T, T)

    elif low_rank2:
        # <hC1 . T . A2 . B2', T > = <hC1 . T . A2, T B2>
        tens1 = nx.dot(hC1, nx.dot(T, hC2[0]))
        tens2 = nx.dot(T, hC2[1])

    else:
        tens1 = nx.dot(hC1, T)
        tens2 = nx.dot(T, hC2.T)
    return nx.sum(constC * T) - nx.sum(tens1 * tens2)


def gwggrad_low_rank_structures(constC, hC1, hC2, T, low_rank1=None, low_rank2=None, nx=None):
    r"""Return the gradient for Gromov-Wasserstein

    
    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, if low_rank1 expects (2, ns, ds) else expects (ns, ns)
        low-rank decomposition of :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, if low_rank2 expects (2, nt, dt) else expects (nt, nt)
        low-rank decomposition of :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`
    low_rank1: bool, optional
        Either hC1 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    low_rank2: bool, optional
        Either hC2 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    grad : array-like, shape (`ns`, `nt`)
        Gromov-Wasserstein gradient

    
    """
    return 2 * tensor_product_low_rank_structures(
        constC, hC1, hC2, T, low_rank1, low_rank2, nx)  # [12] Prop. 2 misses a 2 factor


def init_matrix_semirelaxed_low_rank_structures(
        C1, C2, p, low_rank1=None, low_rank2=None, loss_fun='square_loss', marginal_loss=True, nx=None):
    r"""Return loss matrices and tensors for semi-relaxed Gromov-Wasserstein fast computation

    Returns the value of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` with the
    selected loss function as the loss function of semi-relaxed Gromov-Wasserstein discrepancy.

    Where :

    - :math:`\mathbf{C_1} = \mathbf{A_1} \mathbf{B_1}^\top`: Low-rank decomposition of the metric cost matrix in the source space
    - :math:`\mathbf{C_2}`= \mathbf{A_2} \mathbf{B_2}^\top: Low-rank decomposition of the metric cost matrix in the target space
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
    C1 : array-like, if low_rank1 expects (2, ns, ds) else expects (ns, ns)
        Low-rank decomposition of the metric cost matrix in the source space
    C2 : array-like, if low_rank2 expects (2, nt, dt) else expects (nt, nt)
        Low-rank decomposition of the metric cost matrix in the target space
    p : array-like, shape (ns,)
    low_rank1: bool, optional
        Either hC1 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    low_rank2: bool, optional
        Either hC2 is decomposed as a low-rank tensor or not.
        Default is None and implies a test based on its shape.
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    marginal_loss: bool, optional
        either to compute the constant terms or not
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    hC1 : array-like, shape (2, ns, ds)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (2, nt, dt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    fC2t: array-like, shape (nt, nt)
        :math:`\mathbf{f2}(\mathbf{C2})^\top` matrix in Eq. (6)
    C2_prod: array-like, shape (nt, nt)
        metric cost matrix in the target space C2[0] @ C2[1].T
    C1_prod: array-like, shape (ns, ns), optionally if marginal_loss = True
        metric cost matrix in the source space C1[0] @ C1[1].T
    constC : array-like, shape (ns, nt), optionally if marginal_loss = True
        Constant :math:`\mathbf{C}` matrix in Eq. (6) adapted to srGW


    """
    if nx is None:
        C1, C2, p = list_to_array(C1, C2, p)
        nx = get_backend(C1, C2, p)

    if low_rank1 is None:
        low_rank1 = (len(C1.shape) == 3)

    if low_rank2 is None:
        low_rank2 = (len(C2.shape) == 3)

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        hC1 = nx.copy(C1)

        if low_rank2:
            hC2 = nx.copy(C2)
            hC2[0] = 2 * hC2[0]

            C2_prod = nx.dot(C2[0], C2[1].T)

        else:
            hC2 = 2 * C2
            C2_prod = C2
            
        fC2t = C2_prod ** 2

    elif loss_fun == 'kl_loss':
        def f1(a):
            #return a * nx.log(a + 1e-15) - a
            return a * nx.log(a + 1e-100) - a

        hC1 = nx.copy(C1)
        # warning: hC2 does not preserve the dimensions of C2 with this loss
        if low_rank2:
            C2_prod = nx.dot(C2[0], C2[1].T)
            #hC2 = nx.log(C2_prod + 1e-15)
            hC2 = nx.log(C2_prod + 1e-100)

            fC2t = nx.flip(C2, axis=0)
        else:
            C2_prod = C2
            #hC2 = nx.log(C2_prod + 1e-15)
            hC2 = nx.log(C2_prod + 1e-100)
            fC2t = C2.T
    
    elif loss_fun == 'binary_cross_entropy':
        def f1(a):
            return a * nx.log(a + 1e-100) + (1 - a) * nx.log(1 - a + 1e-100)
        
        hC1 = nx.copy(C1)
        # low_rank2 cannot change the fact that fC2 and hC2 do not admit straight-forward LR decomp.
        if low_rank2:
            C2_prod = nx.dot(C2[0], C2[1].T)
        else:
            C2_prod = C2
            
        fC2t = - nx.log(1 - C2_prod.T + 1e-100)
        
        hC2 = nx.log((C2_prod / (1. - C2_prod)) + 1e-100)

    else:
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    if marginal_loss:
        # compute the constant term in the source space
        if low_rank1:
            C1_prod = nx.dot(C1[0], C1[1].T)
        else:
            C1_prod = C1
        constC = nx.dot(nx.dot(f1(C1_prod), nx.reshape(p, (-1, 1))),
                        nx.ones((1, C2.shape[1]), type_as=p))

        return hC1, hC2, fC2t, C2_prod, constC, C1_prod

    else:
        return hC1, hC2, fC2t, C2_prod