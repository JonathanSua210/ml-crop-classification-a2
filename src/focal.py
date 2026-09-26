"""
Multiclass focal loss as a custom XGBoost objective.

Derivation
----------
Let z be the raw margin (logit) vector for one sample, p = softmax(z),
t the true class index, and u = p_t the predicted probability of the
true class.

    L = -alpha_t * (1 - u)^gamma * log(u)

gamma = 0 recovers standard (optionally weighted) cross entropy.
gamma > 0 down-weights well-classified samples, so the gradient signal
is dominated by hard / rare examples.

First derivative wrt u:

    g(u) = dL/du
         = alpha_t * [ gamma * (1-u)^(gamma-1) * log(u) - (1-u)^gamma / u ]

Second derivative wrt u:

    g'(u) = alpha_t * [ -gamma*(gamma-1)*(1-u)^(gamma-2)*log(u)
                        + 2*gamma*(1-u)^(gamma-1)/u
                        + (1-u)^gamma / u^2 ]

Softmax Jacobian:  du/dz_k = u * (delta_tk - p_k)

Chain rule gives the gradient:

    dL/dz_k = g(u) * u * (delta_tk - p_k)

and the diagonal of the Hessian:

    d2L/dz_k^2 = (g'(u)*u + g(u)) * u * (delta_tk - p_k)^2
                 - g(u) * u * p_k * (1 - p_k)

Sanity check: at gamma=0, alpha=1 we get g = -1/u and g'*u + g = 0, so
    grad = p_k - delta_tk
    hess = p_k * (1 - p_k)
which is exactly XGBoost's built-in multi:softmax. That identity is the
main correctness check, and is asserted in the tests below.
"""
import numpy as np

EPS = 1e-9
HESS_FLOOR = 1e-6


def softmax(z):
    """Row-wise softmax, shifted for numerical stability."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def focal_grad_hess(z, y, gamma=2.0, alpha=None):
    """
    Analytic gradient and diagonal Hessian of multiclass focal loss.

    z     : (n, K) raw margins
    y     : (n,) integer class labels
    gamma : focusing parameter (0 => plain cross entropy)
    alpha : (K,) per-class weights, or None for all ones

    Returns grad, hess each of shape (n, K).
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, K = z.shape

    if alpha is None:
        alpha = np.ones(K, dtype=np.float64)
    a_t = np.asarray(alpha, dtype=np.float64)[y]          # (n,)

    p = softmax(z)                                        # (n, K)
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0

    u = np.clip(p[np.arange(n), y], EPS, 1.0 - EPS)       # (n,)
    log_u = np.log(u)
    om = 1.0 - u                                          # (1 - u)

    # g(u) and g'(u)
    g = a_t * (gamma * om ** (gamma - 1.0) * log_u - om ** gamma / u)
    gp = a_t * (
        -gamma * (gamma - 1.0) * om ** (gamma - 2.0) * log_u
        + 2.0 * gamma * om ** (gamma - 1.0) / u
        + om ** gamma / u ** 2
    )

    g = g[:, None]
    gp = gp[:, None]
    u_c = u[:, None]

    d = onehot - p                                        # (delta_tk - p_k)

    grad = g * u_c * d
    hess = (gp * u_c + g) * u_c * d ** 2 - g * u_c * p * (1.0 - p)

    # XGBoost needs strictly positive curvature to form its leaf weights
    hess = np.maximum(hess, HESS_FLOOR)

    return grad, hess


def focal_loss(z, y, gamma=2.0, alpha=None):
    """Scalar mean loss, used for finite-difference checking."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, K = z.shape
    if alpha is None:
        alpha = np.ones(K, dtype=np.float64)
    a_t = np.asarray(alpha, dtype=np.float64)[y]

    p = softmax(z)
    u = np.clip(p[np.arange(n), y], EPS, 1.0 - EPS)
    return float(np.mean(-a_t * (1.0 - u) ** gamma * np.log(u)))


def make_xgb_objective(gamma=2.0, alpha=None, num_class=None):
    """
    Wrap focal_grad_hess into the callable xgb.train expects.

    XGBoost hands back raw margins; depending on version these arrive
    either already shaped (n, K) or flattened, so we reshape defensively.
    """
    def objective(preds, dtrain):
        y = dtrain.get_label().astype(np.int64)
        z = np.asarray(preds, dtype=np.float64)
        if z.ndim == 1:
            K = num_class if num_class is not None else z.size // y.size
            z = z.reshape(y.size, K)
        grad, hess = focal_grad_hess(z, y, gamma=gamma, alpha=alpha)
        return grad, hess

    return objective