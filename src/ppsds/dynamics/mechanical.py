"""Robotic-arm (n-link) dynamics models.

This module holds only **physical models**: the continuous-time equations and their
discretization, with no uncertainty involved. The "nominal drift + additive perturbation"
assumption and the ambiguity sets live in :mod:`ppsds.ambiguity.additive_drift.conic_moment`.

- :class:`TwoLink` — two links, analytical form: the inertia matrix, Coriolis force, and
  gravity all have closed-form expressions, with a full set of physical parameters (mass,
  link length, center of mass, moment of inertia, damping).
- :class:`NLink` — n links, numerically assembled. The stiffness matrix is omitted and the
  Coriolis force is approximated by empirical damping, so it is **not** an exact robotic-arm
  model; its purpose is to supply a nonlinear, strongly coupled nominal drift for DRPP to
  use. It also provides :meth:`NLink.equilibrium_for_pose` and
  :meth:`NLink.linearize_at_discrete` for linearizing around an equilibrium point.

Both provide ``f_continuous(x, u)`` and ``discretize(x, u, dt)`` (fourth-order Runge–Kutta).
"""

from __future__ import annotations

import math

import numpy as np


class TwoLink:
    def __init__(self, params=None):
        p = {
            'm1': 1.0, 'm2': 0.8,
            'l1': 0.6, 'l2': 0.5,
            'lc1': 0.3, 'lc2': 0.25,
            'I1': 0.02, 'I2': 0.015,
            'd1': 0.05, 'd2': 0.04,
            'g': 9.81
        }
        if params:
            p.update(params)
        self.p = p
        self.n = 2
        self.x_dim = 4
        self.u_dim = 2

    def M(self, q):
        p = self.p
        m1, m2 = p['m1'], p['m2']
        l1, lc1, lc2 = p['l1'], p['lc1'], p['lc2']
        I1, I2 = p['I1'], p['I2']
        a1 = I1 + I2 + m2*l1**2 + m1*lc1**2 + m2*lc2**2
        a2 = m2*l1*lc2
        a3 = I2 + m2*lc2**2
        c2 = math.cos(q[1])
        M11 = a1 + 2*a2*c2
        M12 = a3 + a2*c2
        M22 = a3
        return np.array([[M11, M12],[M12, M22]])

    def C(self, q, qd):
        p = self.p
        a2 = p['m2']*p['l1']*p['lc2']
        s2 = math.sin(q[1])
        c11 = -a2 * s2 * qd[1]
        c12 = -a2 * s2 * (qd[0] + qd[1])
        c21 = a2 * s2 * qd[0]
        c22 = 0.0
        return np.array([[c11, c12],[c21, c22]])

    def g_vec(self, q):
        p = self.p
        b1 = (p['m1']*p['lc1'] + p['m2']*p['l1']) * p['g']
        b2 = p['m2']*p['lc2']*p['g']
        return np.array([b1*math.cos(q[0]) + b2*math.cos(q[0]+q[1]),
                         b2*math.cos(q[0]+q[1])])

    def f_continuous(self, x, u):
        q, qd = x[:2], x[2:]
        M = self.M(q)
        C = self.C(q, qd)
        gvec = self.g_vec(q)
        d = np.array([self.p['d1'], self.p['d2']])
        rhs = u - C.dot(qd) - gvec - d*qd
        qdd = np.linalg.solve(M, rhs)
        return np.hstack((qd, qdd))

    def discretize(self, x, u, dt):
        return rk4_step(self.f_continuous, x, u, dt)


def rk4_step(f_cont, x, u, dt):
    """A single fourth-order Runge–Kutta step. ``f_cont(x, u)`` is the continuous-time
    derivative."""
    k1 = f_cont(x, u)
    k2 = f_cont(x + 0.5*dt*k1, u)
    k3 = f_cont(x + 0.5*dt*k2, u)
    k4 = f_cont(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


class NLink:
    """A numerically assembled model of an n-link robotic arm (see the module docstring for
    the caveats).

    Parameters
    ----------
    n : int
        Number of links. State dimension ``2n`` (the first n are joint angles, the last n
        are angular velocities), control dimension ``n`` (joint torques).
    params : dict, optional
        Physical parameters shared by all links: ``m`` / ``l`` / ``lc`` / ``I`` / ``d`` / ``g``.
    """

    def __init__(self, n=3, params=None):
        self.n = n
        self.x_dim = 2*n
        self.u_dim = n
        if params is None:
            params = {}
        self.m = [params.get('m',0.8)]*n
        self.l = [params.get('l',0.5)]*n
        self.lc = [params.get('lc',0.25)]*n
        self.I = [params.get('I',0.01)]*n
        self.d = [params.get('d',0.01)]*n
        self.g = params.get('g',9.81)

    def M_numeric(self, q):
        n = self.n
        M = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if j >= i:
                    M[i,j] += sum(self.m[k] * (self.lc[k]**2) for k in range(j, self.n))
        M = 0.5*(M + M.T) + 1e-6*np.eye(n)
        return M

    def C_numeric(self, q, qd):
        n = self.n
        C = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                C[i,j] = 0.01 * (abs(qd[i]) + abs(qd[j])) * 0.1
        return C

    def g_numeric(self, q):
        n = self.n
        gvec = np.zeros(n)
        for i in range(n):
            height = sum(self.lc[k] * math.cos(sum(q[:k+1])) for k in range(i+1))
            gvec[i] = self.m[i] * self.g * height
        return gvec

    def f_continuous(self, x, u):
        n = self.n
        q, qd = x[:n], x[n:]
        M = self.M_numeric(q)
        C = self.C_numeric(q, qd)
        gvec = self.g_numeric(q)
        dvec = np.array(self.d)
        rhs = u - C.dot(qd) - gvec - dvec*qd
        qdd = np.linalg.solve(M, rhs)
        return np.hstack((qd, qdd))

    def discretize(self, x, u, dt):
        return rk4_step(self.f_continuous, x, u, dt)

    def equilibrium_for_pose(self, q_star=None, qd_star=None):
        """Returns the equilibrium state and the required torque at pose ``q_star``
        (velocities default to 0)."""
        if qd_star is None:
            qd_star = np.zeros(self.n)
        if q_star is None:
            q_star = np.zeros(self.n)
        x_star = np.hstack((q_star, qd_star))
        u_star = self.g_numeric(q_star)
        return x_star, u_star

    def linearize_at_discrete(self, x_star, u_star, dt=0.02, eps=1e-6):
        """Numerically differentiate the discretized dynamics to get ``A`` / ``B`` at
        ``(x_star, u_star)``."""
        x0 = np.asarray(x_star)
        u0 = np.asarray(u_star)
        f0 = self.discretize(x0, u0, dt)
        n, m = self.x_dim, self.u_dim
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            A[:,i] = (self.discretize(x0+dx, u0, dt) - f0) / eps
        for j in range(m):
            du = np.zeros(m)
            du[j] = eps
            B[:,j] = (self.discretize(x0, u0+du, dt) - f0) / eps
        return A, B
