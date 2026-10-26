import numpy as np


# 1) Nuisance-dominated geometry
def synth_nuisance_dominated(n=5000, d_sig=2, d_nuis=50, sigma_u=10.0, noise=0.1, seed=0):
    """
    X = [signal z, nuisance u], with nuisance dominating Euclidean geometry.
    y depends only on z.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, d_sig))
    u = sigma_u * rng.standard_normal((n, d_nuis))
    X = np.concatenate([z, u], axis=1)
    y = np.sin(2.0 * z[:, 0]) + 0.5 * z[:, 1] + noise * rng.standard_normal(n)
    return X, y


# 2) Many-to-one output collapse (noisy circle; y depends on angle modulo symmetries)
def synth_many_to_one_collapse(n=5000, k=6, r_low=0.8, r_high=1.2, noise=0.05, seed=0):
    """
    X lies on a noisy circle; y = cos(k*theta) collapses many distinct X to same y.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    r = rng.uniform(r_low, r_high, size=n)
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    y = np.cos(k * theta) + noise * rng.standard_normal(n)
    return X, y


# 3) Regime switching with overlapping X (two opposing slopes; weak cue in X)
def synth_regime_switching_overlap(n=5000, noise=0.1, cue_strength=0.5, seed=0):
    """
    Overlapping X cloud with two regimes having opposite relationship with x1.
    Regime s drawn with p(s=1|x)=sigmoid(cue_strength*x1) so overlap remains strong.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    x1, x2 = X[:, 0], X[:, 1]

    p = 1.0 / (1.0 + np.exp(-cue_strength * x1))  # sigmoid
    s = (rng.uniform(0.0, 1.0, size=n) < p).astype(np.int64)

    y0 = x1 + 0.2 * x2
    y1 = -x1 + 0.2 * x2
    y = np.where(s == 0, y0, y1) + noise * rng.standard_normal(n)
    return X, y, s  # return regime labels for analysis


# # 4) Heteroscedastic noise on outputs (noise level depends on mean/sign)
# def synth_heteroscedastic(n=5000, x_low=-3.0, x_high=3.0, base=0.05, extra=0.30, noise_seed=0):
#     """
#     1D x with y = sin(x) + sigma(x)*eps, where sigma jumps when sin(x)>0.
#     """
#     rng = np.random.default_rng(noise_seed)
#     x = rng.uniform(x_low, x_high, size=n)
#     mu = np.sin(x)
#     sigma = base + extra * (mu > 0.0).astype(float)
#     y = mu + sigma * rng.standard_normal(n)
#     X = x[:, None]
#     return X, y, sigma  # sigma returned for calibration checks


# 5) Rare but important spikes/events (tails matter; outliers are predictive)
def synth_rare_spikes(n=5000, p_spike=0.03, spike_amp=5.0, spike_width=0.10, noise=0.05, seed=0):
    """
    2D uniform X with mostly linear y, plus rare localized Gaussian bumps (spikes).
    Tail metrics (top |y|) should highlight methods that preserve rare events.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    x1, x2 = X[:, 0], X[:, 1]

    base_y = 0.2 * x1 + 0.2 * x2

    is_spike = rng.uniform(0.0, 1.0, size=n) < p_spike
    centers = rng.uniform(-1.0, 1.0, size=(n, 2))  # per-sample center (only used if spike)
    d2 = np.sum((X - centers) ** 2, axis=1)
    spike = spike_amp * np.exp(-d2 / (2.0 * spike_width**2)) * is_spike.astype(float)

    y = base_y + spike + noise * rng.standard_normal(n)
    return X, y, is_spike.astype(np.int64)
