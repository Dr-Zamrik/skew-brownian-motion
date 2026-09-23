"""The simulated law against the normal density, doubled and split."""
import numpy as np
import skew as S, skew_plot as P

A, T, H, N = 0.75, 1.0, 0.005, 200_000
r = S.walk(N, alpha=A, t=T, h=H, seed=20260923)
x = r["x"]
fig, ax = P.figure()
edges = np.linspace(-3.5, 3.5, 71)
ax.hist(x, bins=edges, density=True, color=P.TEAL, alpha=0.5, edgecolor="white", lw=0.4,
        label=f"simulation, {N:,} paths")
y = np.linspace(-3.5, 3.5, 700)
ax.plot(y[y > 0], S.density(y[y > 0], T, A), color=P.NAVY, lw=2.4,
        label=r"exact $2\alpha\,\varphi_t$  and  $2(1-\alpha)\,\varphi_t$")
ax.plot(y[y < 0], S.density(y[y < 0], T, A), color=P.NAVY, lw=2.4)
phi = np.exp(-y*y/(2*T))/np.sqrt(2*np.pi*T)
ax.plot(y, phi, color=P.GREY, lw=1.4, ls="--", label=r"Brownian motion, $\varphi_t$")
ax.axvline(0, color=P.GREY, lw=1.2)
ax.annotate(r"mass $\alpha$", xy=(1.15, 0.42), color=P.NAVY, fontsize=12)
ax.annotate(r"mass $1-\alpha$", xy=(-2.25, 0.16), color=P.NAVY, fontsize=12)
ax.set_xlim(-3.5, 3.5)
P.label(ax, x="position at time $t=1$", y="density")
P.legend(ax, loc="upper left")
P.save(fig, "fig_density.png")
