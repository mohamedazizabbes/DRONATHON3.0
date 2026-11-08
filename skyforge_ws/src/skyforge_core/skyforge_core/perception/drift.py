"""Constant-velocity Kalman filter per track, state [x, y, vx, vy].

Floating tracks start with the drift prior v = current + k_leeway * wind and
keep predicting when unseen (covariance grows -> drift ellipse on dashboard).
Land tracks are static: velocity locked to zero.
"""
from __future__ import annotations

import math

import numpy as np


class DriftKF:
    def __init__(self, pos, sigma: float, floating: bool, prior_vel=(0.0, 0.0), q_pos=0.02, q_vel=0.01, max_sigma=30.0,
                 vel_sigma0: float = 0.1, max_dev: float = 0.4, vel_floor: float = 0.0, meas_floor: float = 0.0):
        self.floating = floating
        self.x = np.array([pos[0], pos[1], prior_vel[0] if floating else 0.0, prior_vel[1] if floating else 0.0])
        vel_var = vel_sigma0**2 if floating else 0.0
        self.P = np.diag([sigma**2, sigma**2, vel_var, vel_var])
        self.q_pos, self.q_vel = q_pos, q_vel
        self.max_sigma = max_sigma
        self.prior = tuple(prior_vel) if floating else None
        self.max_dev = max_dev
        self.vel_floor = vel_floor if floating else 0.0
        self.meas_floor = meas_floor if floating else 0.0

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = np.diag([self.q_pos * dt, self.q_pos * dt, self.q_vel * dt, self.q_vel * dt])
        if not self.floating:
            q[2, 2] = q[3, 3] = 0.0
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q
        if not self.floating:
            self.x[2:] = 0.0
            self.P[2:, :] = 0.0
            self.P[:, 2:] = 0.0
        # cap growth so a long-lost track does not explode numerically;
        # scale the whole row/column so P stays positive semi-definite
        s2 = self.max_sigma**2
        for i in (0, 1):
            if self.P[i, i] > s2:
                f = math.sqrt(s2 / self.P[i, i])
                self.P[i, :] *= f
                self.P[:, i] *= f
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z, r_sigma: float) -> None:
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        r_sigma = max(r_sigma, self.meas_floor)
        R = np.eye(2) * r_sigma**2
        y = np.asarray(z) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        if self.floating and self.prior is not None:
            # physically bounded drift: velocity stays near current + leeway
            dvx, dvy = self.x[2] - self.prior[0], self.x[3] - self.prior[1]
            n = math.hypot(dvx, dvy)
            if n > self.max_dev:
                self.x[2] = self.prior[0] + dvx / n * self.max_dev
                self.x[3] = self.prior[1] + dvy / n * self.max_dev
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T  # Joseph form: stays PSD
        self.P = 0.5 * (self.P + self.P.T)
        if self.vel_floor > 0.0:  # never become over-confident about the drift velocity
            for i in (2, 3):
                self.P[i, i] = max(self.P[i, i], self.vel_floor**2)
        if not self.floating:
            self.x[2:] = 0.0

    @property
    def pos(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def vel(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))

    @property
    def sigma(self) -> float:
        return math.sqrt(max(self.P[0, 0], self.P[1, 1], 0.0))

    def predict_pos(self, dt: float) -> tuple[float, float]:
        return (float(self.x[0] + self.x[2] * dt), float(self.x[1] + self.x[3] * dt))

    def mahalanobis2(self, z, r_sigma: float) -> float:
        S = self.P[:2, :2] + np.eye(2) * r_sigma**2
        d = np.asarray(z) - self.x[:2]
        return float(d @ np.linalg.solve(S, d))


def drift_prior(current, wind, k_leeway: float) -> tuple[float, float]:
    return (current[0] + k_leeway * wind[0], current[1] + k_leeway * wind[1])
