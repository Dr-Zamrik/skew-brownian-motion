"""Skew Brownian motion: the simulator and the exact laws it is checked against.

Skew Brownian motion with parameter alpha behaves like Brownian motion away from the
origin; at the origin each excursion chooses the positive side with probability alpha.
Harrison and Shepp's construction is the scaling limit of a lattice walk whose ONLY
asymmetry is a biased step at the single site 0 -- everything here follows from that.
"""
import numpy as np

HOUSE = dict(alpha=0.75, t=1.0, h=0.005, n_paths=200_000)


# ── exact laws ────────────────────────────────────────────────────────────────
def density(y, t, alpha):
    """Transition density from the origin: 2a*phi for y>0, 2(1-a)*phi for y<0."""
    phi = np.exp(-y * y / (2 * t)) / np.sqrt(2 * np.pi * t)
    return np.where(y > 0, 2 * alpha * phi, 2 * (1 - alpha) * phi)


def p_positive(alpha):
    """P(X_t > 0) = alpha, for every t. The sharpest one-number test there is."""
    return alpha


def mean(t, alpha):
    """E[X_t] = (2a-1) E|N(0,t)| = (2a-1) sqrt(2t/pi)."""
    return (2 * alpha - 1) * np.sqrt(2 * t / np.pi)


def second_moment(t):
    """E[X_t^2] = t, whatever alpha is: the skew moves mass, not spread."""
    return t


def mean_occupation(t, alpha):
    """E[time spent in (0,inf) up to t] = alpha * t."""
    return alpha * t


# ── the walk ──────────────────────────────────────────────────────────────────
def walk(n_paths=None, alpha=None, t=None, h=None, seed=0, track_occupation=True):
    """Harrison-Shepp's lattice walk: a fair walk with one biased site.

    From any site but 0 the step is +-h with probability 1/2. From 0 it is +h with
    probability alpha. As h -> 0 this converges to skew Brownian motion; the table in
    the paper measures that convergence rather than asserting it.
    """
    alpha = HOUSE["alpha"] if alpha is None else alpha
    t = HOUSE["t"] if t is None else t
    h = HOUSE["h"] if h is None else h
    n_paths = HOUSE["n_paths"] if n_paths is None else n_paths
    steps = int(round(t / (h * h)))
    rng = np.random.default_rng(seed)
    pos = np.zeros(n_paths, dtype=np.int64)
    occ = np.zeros(n_paths, dtype=np.int64)
    at_zero = np.zeros(n_paths, dtype=np.int64)   # the lattice's OWN mass at the origin
    for _ in range(steps):
        u = rng.random(n_paths)
        at0 = pos == 0
        up = np.where(at0, u < alpha, u < 0.5)
        pos += np.where(up, 1, -1)
        if track_occupation:
            occ += pos > 0
            at_zero += pos == 0
    return dict(x=pos * h, occupation=occ * h * h, zero_time=at_zero * h * h,
                steps=steps, h=h, alpha=alpha, t=t)


def paths(n, alpha, t, h, seed=0):
    """A few whole trajectories, for the figure and the film."""
    steps = int(round(t / (h * h)))
    rng = np.random.default_rng(seed)
    out = np.zeros((n, steps + 1), dtype=np.int64)
    pos = np.zeros(n, dtype=np.int64)
    for s in range(steps):
        u = rng.random(n)
        up = np.where(pos == 0, u < alpha, u < 0.5)
        pos = pos + np.where(up, 1, -1)
        out[:, s + 1] = pos
    return np.linspace(0, t, steps + 1), out * h
