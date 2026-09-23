# Skew Brownian Motion — the code behind the note

Everything that produces the numbers and figures in **Skew Brownian Motion: One Biased
Site and the Asymmetry It Buys** (`TN-2026-25406484`), published at
<https://zamrik.com/research-items/tn-2026-25406484/>.

Skew Brownian motion is ordinary Brownian motion everywhere except one point: at the
origin each excursion chooses the positive side with probability `alpha` and the negative
side with probability `1 - alpha`. Nothing else is changed — no drift, no varying
diffusivity, no boundary. The whole model is one comparison in one line: `u < alpha`
instead of `u < 0.5`, on the steps that begin at the origin.

## Files

| file | what it does |
|---|---|
| `skew.py` | the lattice simulator, and the exact laws it is checked against |
| `compute_numbers.py` | every number quoted in the note; writes `numbers.json` |
| `fig_paths.py` | paths bending at the interface |
| `fig_density.py` | the simulated law against the normal density, doubled and split |
| `fig_alpha.py` | the residuals, and the second moment that separates skew from drift |
| `skew_plot.py` | house plotting defaults |

## Reproducing

```
python3 compute_numbers.py     # ~70 s, writes numbers.json
python3 fig_paths.py           # each figure script writes its own .png
python3 fig_density.py
python3 fig_alpha.py
```

Requires `numpy` and `matplotlib`. Every figure in the note is regenerated from these
scripts on every build of the paper; none is stored and reused.

## What it should reproduce

At `alpha = 0.75`, `t = 1`, `h = 0.005`, 200,000 paths:

| quantity | simulated | exact |
|---|---|---|
| `P(X_t > 0)`, conditioned off the lattice atom | 0.75023 ± 0.00098 | 0.75 |
| `E[X_t]` | 0.39858 ± 0.00205 | 0.398942 |
| `E[X_t^2]` | 1.00067 ± 0.00317 | 1 |
| occupation of the positive half-line | 0.74995 ± 0.00068 | 0.75 |

## The one place lattice and continuum genuinely differ

The walk puts an atom at the origin that the continuum process has not. It is of order the
lattice spacing, and it belongs to neither side. Conditioning it away, and apportioning
the time it represents by `alpha` — an excursion leaves the origin upward with that
probability — is a *derived* correction, not a fitted one. It tracks across four halvings
of the spacing, with residual error 0.0011, 0.0006, 0.0003, 0.0001.

Watch for parity: at `h = 0.04` the walk takes 625 steps, an odd number, so the origin is
unreachable at the final time and the atom vanishes entirely. That is a property of the
lattice, not of the process.
