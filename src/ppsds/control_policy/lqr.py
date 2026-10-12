"""LQR control policy.

The regulation policy used by the linear experiments: drive the state to a target
``x_d`` by infinite-horizon discrete-time LQR.
"""

import numpy as np
from scipy.linalg import solve_discrete_are

class DLQR:
    def __init__(self, Q, R, A, B):
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = -np.linalg.pinv(self.R + self.B.T @ self.P @ self.B) @ self.B.T @ self.P @ self.A


    def __call__(self, x, x_d):
        x_error = x - x_d

        # Kept for reference: the same gain computed by explicit Riccati backward
        # iteration over a finite horizon N, instead of scipy's solve_discrete_are.
        # N = 200
        # P = [None] * (N + 1)
        # P[N] = self.Q
        # for i in range(N, 0, -1):
        #     P[i-1] = self.Q + self.A.T @ P[i] @ self.A - (self.A.T @ P[i] @ self.B) @ np.linalg.pinv(self.R + self.B.T @ P[i] @ self.B) @ (self.B.T @ P[i] @ self.A)
        # K = [None] * N
        # u = [None] * N
        # for i in range(N):
        #     K[i] = -np.linalg.pinv(self.R + self.B.T @ P[i+1] @ self.B) @ self.B.T @ P[i+1] @ self.A
        #     u[i] = K[i] @ x_error
        # u_star = u[N-1]
        u_star = self.K @ x_error

        return u_star
