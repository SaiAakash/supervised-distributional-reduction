from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
import numpy as np
import torch.distributions as D
import torch.nn.functional as F
from src.utils import plan_color

import geoopt
import matplotlib.pyplot as plt
import math

def minkowski_ip(x, y, keepdim=True):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    if x.shape[0] != y.shape[0]:
        return -x[..., 0][None] * y[..., 0][:, None] + torch.sum(
            x[..., 1:][None] * y[..., 1:][:, None], axis=-1
        )
    else:
        return (-x[..., 0] * y[..., 0])[:, None] + torch.sum(
            x[..., 1:] * y[..., 1:], axis=-1, keepdim=True
        )


def minkowski_ip2(x, y):
    """
    Return a n x m matrix where n and m are the number of batchs of x and y.
    """
    return -x[:, 0][None] * y[:, 0][:, None] + torch.sum(
        x[:, 1:][None] * y[:, 1:][:, None], axis=-1
    )


def lorentz_to_poincare(y, r=1):
    return r * y[..., 1:] / (r + y[..., 0][:, None])


def poincare_to_lorentz(x):
    norm_x = torch.norm(x, dim=-1, keepdim=True)
    return torch.cat([1 + norm_x**2, 2 * x], dim=-1) / (1 - norm_x**2)


def sum_mobius(z, y, r=1):
    ip = torch.sum(z * y, axis=-1)
    y_norm2 = torch.sum(y**2, axis=-1)
    z_norm2 = torch.sum(z**2, axis=-1)
    num = (1 + 2 * r * ip + r * y_norm2)[:, None] * z + (1 - r * z_norm2)[:, None] * y
    denom = 1 + 2 * r * ip + r**2 * z_norm2 * y_norm2
    return num / denom[:, None]


def prod_mobius(r, x):
    norm_x = torch.sum(x**2, axis=-1) ** (1 / 2)
    return torch.tanh(r[:, None] * torch.arctanh(norm_x)) * x / norm_x


def dist_poincare(x, y, r=1):
    num = torch.linalg.norm(x - y, axis=-1) ** 2
    denom = (1 - r * torch.linalg.norm(y, axis=-1) ** 2) * (
        1 - r * torch.linalg.norm(x, axis=-1) ** 2
    )
    frac = num / denom
    return torch.arccosh(1 + 2 * r * frac) / np.sqrt(r)


def dist_poincare2(x, y, r=1):
    num = torch.linalg.norm(x[:, None] - y[None], axis=-1) ** 2
    denom = (1 - r * torch.linalg.norm(y, axis=-1) ** 2)[None] * (
        1 - r * torch.linalg.norm(x, axis=-1) ** 2
    )[:, None]
    frac = num / denom
    return torch.arccosh(1 + 2 * r * frac) / np.sqrt(r)


def parallelTransport(v, x0, x1):
    """
    Transport v\in T_x0 H to u\in T_x1 H by following the geodesics by parallel transport
    """
    n, d = v.shape
    if len(x0.shape) == 1:
        x0 = x0.reshape(-1, d)
    if len(x1.shape) == 1:
        x1 = x1.reshape(-1, d)

    u = v + minkowski_ip(x1, v) * (x0 + x1) / (1 - minkowski_ip(x1, x0))
    return u


def expMap(u, x):
    """
    Project u\in T_x H to the surface
    """

    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    norm_u = minkowski_ip(u, u) ** (1 / 2)
    y = torch.cosh(norm_u) * x + torch.sinh(norm_u) * u / norm_u
    return y


def lambd(x, r=1):
    norm_x = torch.norm(x, dim=-1, keepdim=True)
    return 2 / (1 - r * norm_x**2)


def exp_poincare(v, x):
    """
    Project v\in T_x B to the Poincaré ball
    """
    lx = lambd(x)
    norm_v = torch.norm(v, dim=-1, keepdim=True)

    ch = torch.cosh(torch.clamp(lx * norm_v, min=-20, max=20))
    th = torch.tanh(lx * norm_v)
    normalized_v = v / torch.clamp(norm_v, min=1e-6)
    ip_xv = torch.sum(x * normalized_v, dim=-1, keepdim=True)

    num1 = lx * (1 + ip_xv * th) * x
    num2 = th * normalized_v
    denom = 1 / ch + (lx - 1) + lx * ip_xv * th

    return (num1 + num2) / denom


def log_poincare(v, x, r=1.):
    lx = lambd(x, r)
    print('lambda:', lx, lx.shape)
    mob = sum_mobius(-x, v, r)
    print('mob:', mob, mob.shape)
    norm_mob = torch.clamp(torch.norm(mob, dim=-1, keepdim=True), min=1e-15)
    print('norm_mob :', norm_mob, norm_mob.shape)
    ath = torch.arctanh(math.sqrt(r) * norm_mob)
    num = 2 * ath * mob
    print('num:', num, num.shape)
    denum = math.sqrt(r) * lx * norm_mob
    print('denum:', denum, denum.shape)
    return num / denum[:, None]
    
    
def sampleWrappedNormal(mu, Sigma, n, seed=0):
    device = mu.device

    d = len(mu)
    normal = D.MultivariateNormal(torch.zeros((d - 1,), device=device), Sigma)
    x0 = torch.zeros((1, d), device=device)
    x0[0, 0] = 1

    # Sample in T_x0 H
    torch.manual_seed(seed)
    v_ = normal.sample((n,))
    v = F.pad(v_, (1, 0))

    # Transport to T_\mu H and project on H
    u = parallelTransport(v, x0, mu)
    y = expMap(u, mu)

    return y


def sampleLorentzNormal(n, dim=3, scale=1, device="cpu", seed=0):
    mu = torch.zeros(dim, device=device)
    mu[0] = 1  # the base point has one on its first corrdinate in the Lorentz model
    Sigma0 = scale * torch.eye(dim - 1, dtype=torch.float, device=device)
    return sampleWrappedNormal(mu, Sigma0, n, seed)


# **************** Vizualization part ********************************************************


def add_geodesic_grid(ax: plt.Axes, manifold: geoopt.Stereographic, line_width=0.1):

    # define geodesic grid parameters
    N_EVALS_PER_GEODESIC = 10000
    STYLE = "--"
    COLOR = "gray"
    LINE_WIDTH = line_width

    # get manifold properties
    K = manifold.k.item()
    R = manifold.radius.item()

    # get maximal numerical distance to origin on manifold
    if K < 0:
        # create point on R
        r = torch.tensor((R, 0.0), dtype=manifold.dtype)
        # project point on R into valid range (epsilon border)
        r = manifold.projx(r)
        # determine distance from origin
        max_dist_0 = manifold.dist0(r).item()
    else:
        max_dist_0 = np.pi * R
    # adjust line interval for spherical geometry
    circumference = 2 * np.pi * R

    # determine reasonable number of geodesics
    # choose the grid interval size always as if we'd be in spherical
    # geometry, such that the grid interpolates smoothly and evenly
    # divides the sphere circumference
    n_geodesics_per_circumference = 4 * 6  # multiple of 4!
    n_geodesics_per_quadrant = n_geodesics_per_circumference // 2
    grid_interval_size = circumference / n_geodesics_per_circumference
    if K < 0:
        n_geodesics_per_quadrant = int(max_dist_0 / grid_interval_size)

    # create time evaluation array for geodesics
    if K < 0:
        min_t = -1.2 * max_dist_0
    else:
        min_t = -circumference / 2.0
    t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC)[:, None]

    # define a function to plot the geodesics
    def plot_geodesic(gv):
        ax.plot(*gv.t().numpy(), STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # define geodesic directions
    u_x = torch.tensor((0.0, 1.0))
    u_y = torch.tensor((1.0, 0.0))

    # add origin x/y-crosshair
    o = torch.tensor((0.0, 0.0))
    if K < 0:
        x_geodesic = manifold.geodesic_unit(t, o, u_x)
        y_geodesic = manifold.geodesic_unit(t, o, u_y)
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
    else:
        # add the crosshair manually for the sproj of sphere
        # because the lines tend to get thicker if plotted
        # as done for K<0
        ax.axvline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)
        ax.axhline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # add geodesics per quadrant
    for i in range(1, n_geodesics_per_quadrant):
        i = torch.as_tensor(float(i))
        # determine start of geodesic on x/y-crosshair
        x = manifold.geodesic_unit(i * grid_interval_size, o, u_y)
        y = manifold.geodesic_unit(i * grid_interval_size, o, u_x)

        # compute point on geodesics
        x_geodesic = manifold.geodesic_unit(t, x, u_x)
        y_geodesic = manifold.geodesic_unit(t, y, u_y)

        # plot geodesics
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
        if K < 0:
            plot_geodesic(-x_geodesic)
            plot_geodesic(-y_geodesic)


def plotGrid(ax, lw=0.3):
    manifold = geoopt.PoincareBall(c=1)
    circle = plt.Circle((0, 0), 1, color="k", linewidth=3, fill=False)
    ax.add_patch(circle)
    add_geodesic_grid(ax, manifold, line_width=lw)


def plotPoincareFromLorentz(
    X, T, Y, ax, lw=0.3, cmap=plt.cm.get_cmap("tab10"), size_factor=1, thres=3
):
    plotGrid(ax, lw)
    c, s = plan_color(T, Y)
    proj_poincare = lorentz_to_poincare(X)
    ids = torch.where(T.sum(0) > thres)[0]
    ax.scatter(
        proj_poincare[ids, 0],
        proj_poincare[ids, 1],
        s=s[ids] * size_factor,
        c=c[ids],
        cmap=cmap,
    )
    ax.axis("off")
    ax.axis("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


def plotPoincareFromLorentzImages(
    Z,
    centroid,
    T,
    Y,
    ax,
    zoom=0.2,
    thres=0.005,
    title="",
    size=10,
    lw=0.3,
    color=None,
    cmap=plt.cm.get_cmap("tab10"),
    frame_width=2,
):
    plotGrid(ax, lw)
    proj_poincare = lorentz_to_poincare(Z)
    p = centroid.shape[-1]
    n = centroid.shape[0]
    c, s = plan_color(T, Y)
    ids = torch.where(T.sum(0) > thres)[0]
    ax.scatter(
        proj_poincare[ids, 0],
        proj_poincare[ids, 1],
        cmap=cmap,
        alpha=0.5,
        c=c[ids],
        s=s[ids],
    )
    Xbar_im = centroid.reshape(n, int(p ** (0.5)), int(p ** (0.5)))
    for _, i in enumerate(ids):
        img = Image.fromarray(Xbar_im[i].numpy())

        ab = AnnotationBbox(
            OffsetImage(img, zoom=1.5e-1 * np.sqrt(s[i]) * zoom, cmap="gray"),
            (proj_poincare[i, 0], proj_poincare[i, 1]),
            frameon=True,
        )
        ax.add_artist(ab)

    ax.set_title(f"{title}")
    ax.axis("off")
    ax.axis("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)