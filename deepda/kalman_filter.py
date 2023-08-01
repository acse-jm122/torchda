from typing import Callable

import torch

from . import _GenericTensor

__all__ = "apply_KF", "apply_EnKF"


def apply_KF(
    n_steps: int,
    time_obs: _GenericTensor,
    gap: int,
    M: Callable,
    H: torch.Tensor,
    P0: torch.Tensor,
    R: torch.Tensor,
    x0: torch.Tensor,
    y: torch.Tensor,
    start_time: float = 0.0,
    args: tuple = (None,),
) -> torch.Tensor:
    """
    Implementation of the Kalman Filter (constant P assumption).

    This function applies the Kalman Filter algorithm to estimate
    the state of a dynamic system given noisy measurements.

    Parameters
    ----------
    n_steps : The number of time steps to propagate the state forward.

    nobs : The number of observations/measurement updates.

    time_obs : A 1D array contains the observation times in increasing order.

    gap : The number of time steps between consecutive observations.

    M : The state transition function (process model) that predicts the state
        of the system given the previous state and the time range.
        It should have the signature
        M(x: torch.Tensor, time_range: torch.Tensor, *args) -> torch.Tensor.
        'x' is the state vector, 'time_range' is a 1D tensor of time steps
        to predict the state forward, and '*args' represents
        any additional arguments required by the state transition function.

    H : The measurement matrix. A 2D tensor of shape
        (measurement_dim, state_dim),
        where 'measurement_dim' is the dimension of measurement
        and 'state_dim' is the dimension of the state vector.
        This matrix maps the state space to the measurement space.

    P0 : The initial covariance matrix of the state estimate. A 2D tensor of
        shape (state_dim, state_dim).
        It represents the uncertainty of the initial state estimate.

    R : The measurement noise covariance matrix. A 2D tensor of shape
        (measurement_dim, measurement_dim).
        It models the uncertainty in the measurements.

    x0 : The initial state estimate. A 1D tensor of shape (state_dim).

    y : The observed measurements. A 2D tensor of shape
        (measurement_dim, number of observations).
        Each column represents a measurement at a specific time step.

    start_time : The starting time of the filtering process. Default is 0.0.

    args : Additional arguments to pass to the state transition function 'M'.
        Default is (None,).

    Returns
    -------
    x_estimates : A 2D tensor of shape (state_dim, n_steps + 1).
        Each column represents the estimated state vector
        at a specific time step, including the initial state.

    Raises
    ------
    TypeError
        If 'M' is not a callable or 'H' is not a torch.Tensor.

    Notes
    -----
    - The function assumes that the input tensors are properly shaped
        and valid for the Kalman Filter. Ensure that 'x0', 'P0', 'R',
        and 'y' are appropriate for the dimensions of 'M' and 'H'.
    - The function assumes that 'time_obs' contains time points
        that are increasing, and 'gap' specifies the number of time steps
        between consecutive observations.
    - The implementation assumes a constant P assumption,
        meaning the state estimate covariance matrix 'P' remains the same
        throughout the filtering process. If a time-varying 'P' is
        required, you need to modify the function accordingly.
    """
    if not isinstance(M, Callable):
        raise TypeError(
            "`M` must be a callable type in Kalman Filter, "
            f"but given {type(H)=}"
        )
    if not isinstance(H, torch.Tensor):
        raise TypeError(
            "`H` must be an instance of Tensor in Kalman Filter, "
            f"but given {type(H)=}"
        )

    device = x0.device
    x_estimates = torch.zeros((x0.size(0), n_steps + 1), device=device)

    # construct initial state
    xp = x0

    current_time = start_time
    for iobs, time_obs_iobs in enumerate(time_obs):
        istart = iobs * gap
        istop = istart + gap + 1

        # predict
        time_fw = torch.linspace(
            current_time, time_obs_iobs, gap + 1, device=device
        )
        Xf = M(xp, time_fw, *args)

        # update
        xp = Xf[:, -1]
        K = (H @ P0 @ H.T) + R
        w = torch.linalg.solve(K, y[:, iobs] - (H @ xp))
        xp = xp + (P0 @ H.T @ w)

        # store estimates
        x_estimates[:, istart:istop] = Xf
        current_time = time_obs_iobs

    return x_estimates


def apply_EnKF(
    n_steps: int,
    time_obs: _GenericTensor,
    gap: int,  # number of time steps between each observation
    Ne: int,
    M: Callable,
    H: torch.Tensor | Callable,
    P0: torch.Tensor,
    R: torch.Tensor,
    x0: torch.Tensor,
    y: torch.Tensor,
    start_time: float = 0.0,
    args: tuple = (None,),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of the Ensemble Kalman Filter
    See e.g. Evensen, Ocean Dynamics (2003), Eqs. 44--54
    """
    if not isinstance(M, Callable):
        raise TypeError(
            "`M` must be a callable type in Ensemble Kalman Filter, "
            f"but given {type(H)=}"
        )
    if not isinstance(H, (Callable, torch.Tensor)):
        raise TypeError(
            "`H` must be a callable type or an instance of Tensor "
            f"in Ensemble Kalman Filter, but given {type(H)=}"
        )
    if isinstance(M, torch.nn.Module) or isinstance(H, torch.nn.Module):
        from warnings import warn

        warn(
            "This EnKF function cannot calculate steady output for "
            "nueral network `M` or `H`, it would be fixed in the future.",
            FutureWarning,
        )

    device = x0.device
    x_dim, y_dim = x0.size(0), y.size(0)
    x_ave = torch.zeros((x_dim, n_steps + 1), device=device)
    x_ens = torch.zeros((x_dim, n_steps + 1, Ne), device=device)

    # construct initial ensemble
    X = (
        torch.distributions.MultivariateNormal(
            loc=x0.ravel(), covariance_matrix=P0
        )
        .sample([Ne])
        .T
    ).to(device=device)
    one_over_Ne = 1.0 / Ne
    one_over_Ne_minus_one = 1.0 / (Ne - 1.0)

    current_time = start_time
    running_mean = torch.empty((x_dim, gap + 1), device=device)
    for iobs, time_obs_iobs in enumerate(time_obs):
        istart = iobs * gap
        istop = istart + gap + 1
        running_mean.zero_()
        time_fw = torch.linspace(
            current_time, time_obs_iobs, gap + 1, device=device
        )
        for e in range(Ne):
            # prediction phase for each ensemble member
            Xf = M(X[:, e], time_fw, *args)
            x_ens[:, istart:istop, e] = Xf
            X[:, e] = Xf[:, -1]
            running_mean = running_mean + Xf
        # Noise the obs (Burgers et al, 1998)
        D = (
            torch.distributions.MultivariateNormal(
                loc=y[:, iobs].ravel(),
                covariance_matrix=R,
            )
            .sample([Ne])
            .T
        ).to(device=device)
        E = torch.mean(X, dim=1).view((-1, 1))
        if isinstance(H, Callable):
            Xh = H(X)
            Xh_minus_z_mean = Xh - torch.mean(Xh, dim=1).view((-1, 1))
            Pzz = (
                one_over_Ne_minus_one * (Xh_minus_z_mean @ Xh_minus_z_mean.T)
                + R
            )
            Pxz = one_over_Ne_minus_one * ((X - E) @ Xh_minus_z_mean.T)
            X = X + Pxz @ torch.linalg.solve(Pzz, D - Xh)
        else:  # isinstance(H, torch.Tensor)
            A = X - E
            Pe = one_over_Ne_minus_one * (A @ A.T)
            # Assembly of the Kalman gain matrix
            K = (H @ Pe @ H.T) + R
            # Solve
            w = torch.linalg.solve(K, D - (H @ X))
            # Update
            X = X + (Pe @ H.T @ w)
        running_mean = one_over_Ne * running_mean
        x_ave[:, istart:istop] = running_mean
        current_time = time_obs_iobs

    return x_ave, x_ens
