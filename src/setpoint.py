"""Set-point modes for the pendulum, written in error coordinates z = x - x*_m.

Mode m stabilizes x*_m = (theta*_m, 0) with the gravity-compensating input
u*_m = -m g l sin(theta*_m)  (from  theta_ddot = g/l sin(theta) + u/(m l^2) = 0).
In z-coordinates every mode has its equilibrium at z = 0, so one Lyapunov "shape" V(z),
one box and one sublevel set {V <= rho} can be shared by all modes, and the training loop
and certificate format of Yang et al. (2024) apply unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.pendulum as pendulum

MASS, LENGTH, DAMPING, G = 0.15, 0.5, 0.1, 9.81


def setpoint_equilibrium(theta_star: float, mass: float = MASS, length: float = LENGTH):
    return [theta_star, 0.0], [-mass * G * length * math.sin(theta_star)]


class ErrorCoordDynamics(nn.Module):
    """z+ = f(z + x*, u) - x*  for a discrete-time pendulum."""

    def __init__(self, theta_star: float, dt: float = 0.05, mass: float = MASS):
        super().__init__()
        cont = pendulum.PendulumDynamics(m=mass, l=LENGTH, beta=DAMPING)
        self.base = dynamical_system.SecondOrderDiscreteTimeSystem(
            cont, dt=dt,
            position_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
            velocity_integration=dynamical_system.IntegrationMethod.ExplicitEuler)
        x_eq, u_eq = setpoint_equilibrium(theta_star, mass)
        self.register_buffer("x_star", torch.tensor(x_eq), persistent=False)
        self.register_buffer("u_star", torch.tensor(u_eq), persistent=False)
        self.nx, self.nu = 2, 1

    def forward(self, z, u):
        return self.base(z + self.x_star, u) - self.x_star

    @property
    def x_equilibrium(self):
        return torch.zeros(2, device=self.x_star.device)

    @property
    def u_equilibrium(self):
        return self.u_star
