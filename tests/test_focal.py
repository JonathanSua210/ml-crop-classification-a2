"""Verify the analytic focal-loss derivatives against finite differences."""
import numpy as np
from src.focal import softmax, focal_grad_hess, focal_loss

rng = np.random.default_rng(0)


def numeric_grad_hess(z, y, gamma, alpha, h=1e-5):
    """Central-difference gradient and diagonal Hessian of the SUM loss."""
    n, K = z.shape
    g = np.zeros_like(z)
    H = np.zeros_like(z)
    for i in range(n):
        for k in range(K):
            zp = z.copy(); zp[i, k] += h
            zm = z.copy(); zm[i, k] -= h
            # focal_loss returns the mean, so multiply back up to a sum
            lp = focal_loss(zp, y, gamma, alpha) * n
            lm = focal_loss(zm, y, gamma, alpha) * n
            l0 = focal_loss(z, y, gamma, alpha) * n
            g[i, k] = (lp - lm) / (2 * h)
            H[i, k] = (lp - 2 * l0 + lm) / (h ** 2)
    return g, H


def check(gamma, alpha, tag):
    n, K = 6, 4
    z = rng.normal(0, 1.5, size=(n, K))
    y = rng.integers(0, K, size=n)

    ga, ha = focal_grad_hess(z, y, gamma=gamma, alpha=alpha)
    gn, hn = numeric_grad_hess(z, y, gamma, alpha)

    # only compare hessian entries that were not clipped by the floor
    mask = ha > 1e-6 + 1e-12

    gerr = np.abs(ga - gn).max()
    herr = np.abs(ha[mask] - hn[mask]).max()

    print(f"{tag}")
    print(f"   max |grad analytic - grad numeric| = {gerr:.3e}")
    print(f"   max |hess analytic - hess numeric| = {herr:.3e}  "
          f"({mask.sum()}/{mask.size} entries unclipped)")
    return gerr, herr


print("=== finite-difference verification ===\n")
check(0.0, None, "gamma=0, alpha=None (should reduce to cross entropy)")
check(2.0, None, "gamma=2, alpha=None")
check(2.0, np.array([1.0, 3.0, 0.5, 2.0]), "gamma=2, weighted alpha")
check(1.0, None, "gamma=1, alpha=None")

print("\n=== identity check: gamma=0 must equal softmax cross entropy ===")
n, K = 5, 4
z = rng.normal(0, 1.0, size=(n, K))
y = rng.integers(0, K, size=n)
p = softmax(z)
onehot = np.zeros_like(p)
onehot[np.arange(n), y] = 1.0

ga, ha = focal_grad_hess(z, y, gamma=0.0, alpha=None)
grad_builtin = p - onehot
hess_builtin = p * (1 - p)

print(f"max |grad - (p - onehot)|   = {np.abs(ga - grad_builtin).max():.3e}")
print(f"max |hess - p(1-p)|         = "
      f"{np.abs(ha - hess_builtin)[hess_builtin > 1e-6].max():.3e}")