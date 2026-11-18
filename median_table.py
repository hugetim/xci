"""Median-centred beta for XCI v0.1: Beta(a, kappa - a) whose median is mu, i.e. I_mu(a, kappa - a) = 1/2 (I = the
regularized incomplete beta, scipy.special.betainc). The median rises monotonically with a, so the root is unique.

Solving for the root inside the model at every gradient evaluation would be slow, so the model reads a from a table.
z = logit(a / kappa) is smooth in u = logit(mu) and w = log(kappa). The exact roots (solve_median) are computed on a
fine (u, w) grid, a bicubic spline through them gives z and its derivatives at the knots, and herm_a interpolates
between knots by bicubic Hermite interpolation. That is plain array arithmetic, so it runs inside PyTensor's compiled
graph and PyTensor differentiates it exactly. The table matches the exact roots to within 3e-9 in a.

The table spans logit(mu) to +-27.7, which covers XCI v0.1's clip on mu (1e-12, at logit +-27.63), and kappa from 0.3
to 5,000. It is built once per process (a few seconds).
"""
import numpy as np

TABLE_U = (-27.7, 27.7, 1109)                     # logit(mu): range and number of knots (spacing 0.05)
TABLE_W = (np.log(0.3), np.log(5000.0), 197)      # log(kappa)
H = 1e-5  # relative step for the central differences of betainc in a and b


def solve_logit_centre(mu, kappa, n_iter=25, tol=1e-13):
    """Starting point for solve_median: a with psi(a) - psi(kappa - a) = logit(mu) (the logit-centred beta), by
    Newton's method in v = logit(a / kappa). Close to the median-centred root."""
    from scipy.special import digamma, expit, logit, polygamma
    mu, kappa = np.broadcast_arrays(np.asarray(mu, float), np.asarray(kappa, float))
    L = logit(mu)
    v = L.copy()
    for _ in range(n_iter):
        a = kappa * expit(v); b = kappa - a
        g = digamma(a) - digamma(b) - L
        gp = (polygamma(1, a) + polygamma(1, b)) * a * b / kappa
        step = np.clip(g / gp, -3.0, 3.0)
        v = v - step
        if np.abs(step).max() < tol:
            break
    return kappa * expit(v)


def inc_beta_parts(a, b, x):
    """I_x(a, b) and its derivatives in a and b (central differences)."""
    from scipy.special import betainc
    I = betainc(a, b, x)
    Ia = (betainc(a * (1 + H), b, x) - betainc(a * (1 - H), b, x)) / (2 * H * a)
    Ib = (betainc(a, b * (1 + H), x) - betainc(a, b * (1 - H), x)) / (2 * H * b)
    return I, Ia, Ib


def solve_median(mu, kappa, n_newton=30, n_bisect=200, tol=1e-12):
    """The root a of I_mu(a, kappa - a) = 1/2 (numpy): Newton's method in v = logit(a / kappa), from the logit-centred
    solution; elements that have not converged after the Newton steps are finished by bisection."""
    from scipy.special import betainc, expit, logit
    mu, kappa = np.broadcast_arrays(np.asarray(mu, float), np.asarray(kappa, float))
    v = logit(solve_logit_centre(mu, kappa) / kappa)
    parts = None
    for _ in range(n_newton):
        a = kappa * expit(v); b = kappa - a
        parts = inc_beta_parts(a, b, mu)
        F = parts[0] - 0.5
        if np.abs(F).max() < tol:
            break
        dF = (parts[1] - parts[2]) * a * b / kappa            # dF/dv < 0
        v = v - np.clip(F / np.where(dF < 0, dF, -1e-300), -2.0, 2.0)
        parts = None
    a = kappa * expit(v)
    if parts is None:
        parts = inc_beta_parts(a, kappa - a, mu)
    bad = np.abs(parts[0] - 0.5) >= tol
    if bad.any():  # bisection in v for the rest (F decreases in v)
        m, k = mu[bad], kappa[bad]
        lo, hi = np.full(m.shape, -40.0), np.full(m.shape, 40.0)
        for _ in range(n_bisect):
            mid = (lo + hi) / 2; f = betainc(k * expit(mid), k - k * expit(mid), m) - 0.5
            lo, hi = np.where(f > 0, mid, lo), np.where(f > 0, hi, mid)
        v = v.copy(); v[bad] = (lo + hi) / 2; a = kappa * expit(v)
    return a


_HERM = None


def _herm_tables():
    """Knot values of z and its derivatives z_u, z_w, z_uw, from a bicubic spline through the exact roots."""
    global _HERM
    if _HERM is None:
        from scipy.interpolate import RectBivariateSpline
        from scipy.special import expit, logit
        u = np.linspace(*TABLE_U); w = np.linspace(*TABLE_W)
        U, W = np.meshgrid(u, w, indexing="ij")
        a = solve_median(expit(U.ravel()), np.exp(W.ravel())).reshape(U.shape)
        s = RectBivariateSpline(u, w, logit(a / np.exp(W)), kx=3, ky=3, s=0)
        T = [s(u, w, dx=dx, dy=dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))]
        _HERM = dict(u0=u[0], w0=w[0], hu=u[1] - u[0], hw=w[1] - w[0], nu=len(u), nw=len(w),
                     Z=T[0].ravel(), Zu=T[1].ravel(), Zw=T[2].ravel(), Zuw=T[3].ravel())
    return _HERM


def herm_z(u, w, xp=np):
    """z(u, w) by bicubic Hermite interpolation; u, w outside the table are clamped to its edges."""
    H = _herm_tables()
    if xp is np:
        floor, clip, take, cast = np.floor, np.clip, (lambda T, i: T[i]), (lambda x: x.astype(int))
    else:
        import pytensor.tensor as pt
        floor, clip, cast = pt.floor, pt.clip, (lambda x: pt.cast(x, "int64"))
        take = lambda T, i: pt.as_tensor_variable(T)[i]
    hu, hw = H["hu"], H["hw"]
    u = clip(u, H["u0"], H["u0"] + hu * (H["nu"] - 1)); w = clip(w, H["w0"], H["w0"] + hw * (H["nw"] - 1))
    fi = clip(floor((u - H["u0"]) / hu), 0, H["nu"] - 2); fj = clip(floor((w - H["w0"]) / hw), 0, H["nw"] - 2)
    t = (u - H["u0"]) / hu - fi; s = (w - H["w0"]) / hw - fj
    i, j = cast(fi), cast(fj)
    h0 = lambda x: (2 * x - 3) * x * x + 1; h1 = lambda x: ((x - 2) * x + 1) * x     # value / slope basis at x = 0
    g0 = lambda x: (3 - 2 * x) * x * x; g1 = lambda x: (x - 1) * x * x                 # value / slope basis at x = 1
    bu = ((h0(t), h1(t)), (g0(t), g1(t))); bw = ((h0(s), h1(s)), (g0(s), g1(s)))
    z = 0
    for di in (0, 1):
        for dj in (0, 1):
            k = (i + di) * H["nw"] + (j + dj)
            z = z + (take(H["Z"], k) * bu[di][0] * bw[dj][0] + hu * take(H["Zu"], k) * bu[di][1] * bw[dj][0]
                     + hw * take(H["Zw"], k) * bu[di][0] * bw[dj][1] + hu * hw * take(H["Zuw"], k) * bu[di][1] * bw[dj][1])
    return z


def herm_a(mu, kappa, xp=np):
    """Median-centred a(mu, kappa) from the table (numpy, or PyTensor with xp=pytensor.tensor)."""
    if xp is np:
        from scipy.special import expit
        mu, kappa = np.asarray(mu, float), np.asarray(kappa, float)
        return kappa * expit(herm_z(np.log(mu) - np.log1p(-mu), np.log(kappa)))
    import pytensor.tensor as pt
    return kappa * pt.sigmoid(herm_z(pt.log(mu) - pt.log1p(-mu), pt.log(kappa), xp=pt))
