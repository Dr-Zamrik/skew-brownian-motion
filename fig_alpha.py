"""How exactly the parameter is recovered, and what separates skew from drift."""
import numpy as np
import skew as S, skew_plot as P

T, H, N = 1.0, 0.01, 40_000
AS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
res_p, res_o, se_p, sec, se_s = [], [], [], [], []
for i, a in enumerate(AS):
    r = S.walk(N, alpha=a, t=T, h=H, seed=7 + 100*i)
    x, pz = r["x"], np.mean(r["x"] == 0)
    p = (x > 0).astype(float) / (1 - pz)
    res_p.append(p.mean() - a); se_p.append(p.std(ddof=1)/np.sqrt(N))
    res_o.append((r["occupation"] + a*r["zero_time"]).mean() - a*T)
    sec.append((x**2).mean()); se_s.append((x**2).std(ddof=1)/np.sqrt(N))

fig, (ax1, ax2) = P.panels(2)
# LEFT: the residual, which is what a diagonal hides
ax1.axhline(0, color=P.GREY, lw=1.4, ls="--", label="exact")
ax1.errorbar(AS, res_p, yerr=se_p, fmt="o", ms=6, color=P.NAVY, ecolor=P.NAVY, capsize=3,
             lw=1.2, label=r"$\mathbb{P}(X_t>0) - \alpha$")
ax1.plot(AS, res_o, "s", ms=6, color=P.TEAL, label=r"occupation $-\;\alpha t$")
ax1.set_ylim(-0.006, 0.006); ax1.set_xlim(0, 1)
P.label(ax1, x=r"skew parameter $\alpha$", y="simulated minus exact")
P.legend(ax1, loc="upper left")

# RIGHT: the second moment is blind to the skew; a drift with the same mean is not
g = np.linspace(0, 1, 200)
ax2.plot(g, np.ones_like(g)*T, color=P.NAVY, lw=2.4, label=r"skew: $\mathbb{E}[X_t^2]=t$, any $\alpha$")
ax2.plot(g, T + (2*g - 1)**2 * (2*T/np.pi), color=P.TEAL, lw=2.0, ls="--",
         label="a drift with the same mean")
ax2.errorbar(AS, sec, yerr=se_s, fmt="o", ms=6, color=P.NAVY, ecolor=P.NAVY, capsize=3,
             lw=1.2, label="simulated")
ax2.set_xlim(0, 1)
P.label(ax2, x=r"skew parameter $\alpha$", y="second moment")
P.legend(ax2, loc="upper center")
P.save(fig, "fig_alpha.png")
