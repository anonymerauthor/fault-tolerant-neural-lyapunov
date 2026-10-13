"""Model factory loaded by alpha-beta-CROWN (Customized(...)) for one pendulum mode."""
import torch.nn as nn  # noqa: F401  (referenced by eval() in config strings)

import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.models as models
import neural_lyapunov_training.pendulum as pendulum


def create_pendulum_setpoint(theta_star: float, dt: float = 0.05, **kwargs):
    """Set-point mode in error coordinates z = x - x* (see src/setpoint.py)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from setpoint import ErrorCoordDynamics
    return models.create_model(ErrorCoordDynamics(theta_star, dt=dt), **kwargs)


def create_pendulum_mode(mass: float, dt: float = 0.05, **kwargs):
    return models.create_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=mass, l=0.5, beta=0.1),
            dt=dt,
            position_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
            velocity_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
        ),
        **kwargs,
    )
