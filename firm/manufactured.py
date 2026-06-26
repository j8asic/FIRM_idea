"""
Manufactured fields with analytic value / gradient / Laplacian, plus analytic
boundary-condition data (wall Neumann flux, free-surface target value).

All callables accept a point array ``P`` of shape (..., d) and return:
  value(P) -> (...)
  grad(P)  -> (..., d)
  laplacian(P) -> (...)
so they work for a single point (d,) or a cloud (N, d) by broadcasting.
"""
import numpy as np


class Field:
    def __init__(self, name, value, grad, laplacian):
        self.name = name
        self.value = value
        self.grad = grad
        self.laplacian = laplacian

    def wall_flux(self, foot, normal):
        """Neumann data g = grad(p)·n evaluated at the wall foot point."""
        return float(np.dot(self.grad(np.asarray(foot, float)), normal))


def linear_field(a, b=0.0, name="lin"):
    a = np.asarray(a, float)

    def value(P):
        P = np.asarray(P, float)
        return P @ a + b

    def grad(P):
        P = np.asarray(P, float)
        return np.broadcast_to(a, P.shape).copy()

    def laplacian(P):
        P = np.asarray(P, float)
        return np.zeros(P.shape[:-1])

    return Field(name, value, grad, laplacian)


def quadratic_field(Q, a, b=0.0, name="quad"):
    Q = np.asarray(Q, float)
    Q = 0.5 * (Q + Q.T)  # symmetrize
    a = np.asarray(a, float)
    trQ = float(np.trace(Q))

    def value(P):
        P = np.asarray(P, float)
        return 0.5 * np.einsum("...i,ij,...j->...", P, Q, P) + P @ a + b

    def grad(P):
        P = np.asarray(P, float)
        return P @ Q + a  # Q symmetric -> (Q x)_k

    def laplacian(P):
        P = np.asarray(P, float)
        return np.full(P.shape[:-1], trQ)

    return Field(name, value, grad, laplacian)


def trig_field(k=np.pi, name="trig"):
    """sin(k x) cos(k y) in 2D;  laplacian = -2 k^2 * value."""

    def value(P):
        P = np.asarray(P, float)
        return np.sin(k * P[..., 0]) * np.cos(k * P[..., 1])

    def grad(P):
        P = np.asarray(P, float)
        gx = k * np.cos(k * P[..., 0]) * np.cos(k * P[..., 1])
        gy = -k * np.sin(k * P[..., 0]) * np.sin(k * P[..., 1])
        return np.stack([gx, gy], axis=-1)

    def laplacian(P):
        return -2.0 * k * k * value(P)

    return Field(name, value, grad, laplacian)


def complex_field(k=np.pi, alpha=0.3, name="complex"):
    """Capstone field: sin(k x) cos(k y) + alpha (x^2 + y^2).

    laplacian = -2 k^2 sin(k x) cos(k y) + 4 alpha  (sign-varying source).
    """

    def value(P):
        P = np.asarray(P, float)
        x, y = P[..., 0], P[..., 1]
        return np.sin(k * x) * np.cos(k * y) + alpha * (x * x + y * y)

    def grad(P):
        P = np.asarray(P, float)
        x, y = P[..., 0], P[..., 1]
        gx = k * np.cos(k * x) * np.cos(k * y) + 2.0 * alpha * x
        gy = -k * np.sin(k * x) * np.sin(k * y) + 2.0 * alpha * y
        return np.stack([gx, gy], axis=-1)

    def laplacian(P):
        P = np.asarray(P, float)
        x, y = P[..., 0], P[..., 1]
        return -2.0 * k * k * np.sin(k * x) * np.cos(k * y) + 4.0 * alpha

    return Field(name, value, grad, laplacian)


def franke_field(name="franke"):
    """Franke's bivariate test function on [0,1]^2 (Franke 1980; the 2017 paper's
    operator-error field). Two Gaussian peaks, a sharp dip and a sloping term, with
    analytic gradient and Laplacian. Each term is A*exp(E) with
    E = -ax*(9x-cx)^2 - ay*(9y-cy)^2  (term 2 is linear in y: -(9y+1)/10)."""
    # (A, ax, cx, ay, cy, y_linear?)
    terms = [
        (0.75, 1.0 / 4.0, 2.0, 1.0 / 4.0, 2.0, False),
        (0.75, 1.0 / 49.0, -1.0, 1.0 / 10.0, -1.0, True),   # y term is linear: -(9y+1)/10
        (0.50, 1.0 / 4.0, 7.0, 1.0 / 4.0, 3.0, False),
        (-0.20, 1.0, 4.0, 1.0, 7.0, False),
    ]

    def _parts(P):
        P = np.asarray(P, float)
        x, y = P[..., 0], P[..., 1]
        out = []
        for A, ax, cx, ay, cy, ylin in terms:
            u = 9.0 * x - cx
            E = -ax * u * u
            Ex = -2.0 * ax * u * 9.0
            Exx = -2.0 * ax * 81.0
            if ylin:
                v = 9.0 * y - cy                     # cy = -1 -> v = 9y+1
                E = E - ay * v                        # -(1/10)(9y+1)
                Ey = -ay * 9.0
                Eyy = 0.0
            else:
                v = 9.0 * y - cy
                E = E - ay * v * v
                Ey = -2.0 * ay * v * 9.0
                Eyy = -2.0 * ay * 81.0
            T = A * np.exp(E)
            out.append((T, Ex, Ey, Exx, Eyy))
        return out

    def value(P):
        return sum(T for T, *_ in _parts(P))

    def grad(P):
        gx = sum(T * Ex for T, Ex, Ey, Exx, Eyy in _parts(P))
        gy = sum(T * Ey for T, Ex, Ey, Exx, Eyy in _parts(P))
        return np.stack([gx, gy], axis=-1)

    def laplacian(P):
        return sum(T * (Ex * Ex + Exx + Ey * Ey + Eyy) for T, Ex, Ey, Exx, Eyy in _parts(P))

    return Field(name, value, grad, laplacian)


# Convenient default instances --------------------------------------------------
LIN2D = linear_field([0.37, -0.52], 0.13)
QUAD2D = quadratic_field([[1.3, 0.4], [0.4, -0.7]], [0.2, -0.1], 0.05)
TRIG2D = trig_field(np.pi)
COMPLEX2D = complex_field(np.pi, 0.3)

REGISTRY = {f.name: f for f in (LIN2D, QUAD2D, TRIG2D, COMPLEX2D)}
