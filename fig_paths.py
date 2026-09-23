"""Paths bending at the interface: the same walk, fair everywhere but one site."""
import numpy as np
import skew as S, skew_plot as P

A, T, H = 0.75, 1.0, 0.002
fig, (ax1, ax2) = P.panels(2)
for ax, a, col, name in ((ax1, 0.5, P.NAVY, r"$\alpha = 0.5$ — ordinary Brownian motion"),
                         (ax2, A,  P.TEAL, r"$\alpha = 0.75$ — skewed at the origin")):
    ts, X = S.paths(14, a, T, H, seed=5)
    for i in range(X.shape[0]):
        ax.plot(ts, X[i], color=col, lw=0.8, alpha=0.65)
    ax.axhline(0.0, color=P.GREY, lw=1.6)
    ax.set_ylim(-2.6, 2.6); ax.set_xlim(0, T)
    ax.set_title(name, color=P.GREY, fontsize=12.5)
    frac = np.mean(X[:, -1] > 0)
    ax.annotate(f"{int(round(frac*100))} per cent finish above", xy=(0.03, -2.35),
                fontsize=11, color=P.GREY)
    P.label(ax, x="time", y="position" if ax is ax1 else None)
P.save(fig, "fig_paths.png")
