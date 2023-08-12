"""
`apply_EnKF` implementation modified from
https://github.com/MagneticEarth/book.magneticearth.org/blob/main/data_assimilation/myEnKF.py
'Copyright 2021 Ashley Smith'

Also reference to FilterPy library.
http://github.com/rlabbe/filterpy
'Copyright 2015 Roger R Labbe Jr.'

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
"""
from typing import Callable

import torch
from torch.func import jacfwd, jacrev

from . import _GenericTensor


@torch.no_grad()
def apply_KF(
    time_obs: _GenericTensor,
    gaps: _GenericTensor,
    M: Callable,
    H: torch.Tensor | Callable,
    P0: torch.Tensor,
    R: torch.Tensor,
    x0: torch.Tensor,
    y: torch.Tensor,
    start_time: float = 0.0,
    args: tuple = (None,),
) -> torch.Tensor:
    r"""
    Implementation of the Kalman Filter (constant P assumption).

    This function applies the Kalman Filter algorithm to estimate
    the state of a dynamic system given noisy measurements. It is executed
    within a no-grad context, meaning that gradient computation is disabled.

    Parameters
    ----------
    time_obs : _GenericTensor
        A 1D array containing the observation times in increasing order.

    gaps : _GenericTensor
        A 1D array containing the number of time steps
        between consecutive observations.

    M : Callable
        The state transition function (process model) that predicts the state
        of the system given the previous state and the time range.
        It should have the signature
        M(x: torch.Tensor, time_range: torch.Tensor, \*args) -> torch.Tensor.
        'x' is the state vector, 'time_range' is a 1D tensor of time steps
        to predict the state forward, and '\*args' represents
        any additional arguments required by the state transition function.

    H : torch.Tensor | Callable
        The measurement matrix or a function that
        computes the measurement matrix. If 'H' is a torch.Tensor,
        it is a 2D tensor of shape (measurement_dim, state_dim),
        where 'measurement_dim' is the dimension of measurement
        and 'state_dim' is the dimension of the state vector.
        This matrix maps the state space to the measurement space.
        If 'H' is a Callable, it should have the signature
        H(x: torch.Tensor) -> torch.Tensor to compute the measurement,
        and 'H' must be able to handle the input 'x' with shape
        (state_dim,). The output of Callable 'H' must be a Tensor with shape
        (measurement_dim,).

    P0 : torch.Tensor
        The initial covariance matrix of the state estimate. A 2D tensor of
        shape (state_dim, state_dim).
        It represents the uncertainty of the initial state estimate.

    R : torch.Tensor
        The measurement noise covariance matrix. A 2D tensor of shape
        (measurement_dim, measurement_dim).
        It models the uncertainty in the measurements.

    x0 : torch.Tensor
        The initial state estimate. A 1D tensor of shape (state_dim).

    y : torch.Tensor
        The observed measurements. A 2D tensor of shape
        (number of observations, measurement_dim).
        Each row represents a measurement at a specific time step.

    start_time : float, optional
        The starting time of the filtering process. Default is 0.0.

    args : tuple, optional
        Additional arguments to pass to the state transition function 'M'.
        Default is (None,).

    Returns
    -------
    x_estimates : A 2D tensor of shape (num_steps + 1, state_dim).
        Each row represents the estimated state vector
        at a specific time step, including the initial state.

    Raises
    ------
    TypeError
        If 'M' is not a callable or 'H' is not a torch.Tensor or Callable.

    Notes
    -----
    - The function assumes that the input tensors are properly shaped
      and valid for the Kalman Filter. Ensure that 'x0', 'P0', 'R',
      and 'y' are appropriate for the dimensions of 'M' and 'H'.
    - The function assumes that 'time_obs' contains time points
      that are increasing, and 'gaps' specifies each number of time steps
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
    if not isinstance(H, (Callable, torch.Tensor)):
        raise TypeError(
            "`H` must be a callable type or an instance of Tensor "
            f"in Kalman Filter, but given {type(H)=}"
        )
    if isinstance(H, torch.nn.Module):
        from warnings import warn

        warn(
            "Jacobian calculation on Nerual Network `H` might be incorrect "
            "for the current Kalman Filter implementation.",
            FutureWarning,
        )

    device = x0.device
    x_dim = x0.numel()
    x_estimates = torch.zeros((int(sum(gaps)) + 1, x_dim), device=device)

    if isinstance(H, Callable):
        jacobian = jacrev(H) if x0.numel() >= y[0].numel() else jacfwd(H)

    # construct initial state
    x = x0.ravel()

    current_time = start_time
    for iobs, (time_obs_iobs, gap) in enumerate(zip(time_obs, gaps)):
        istart = iobs * gap
        istop = istart + gap + 1

        # predict
        time_fw = torch.linspace(
            current_time, time_obs_iobs, gap + 1, device=device
        )
        X = M(x, time_fw, *args)

        # update
        x = X[-1]
        H_mat = jacobian(x) if isinstance(H, Callable) else H
        K = (H_mat @ P0 @ H_mat.T) + R
        w = torch.linalg.solve(K, y[iobs] - (H_mat @ x))
        x = x + (P0 @ H_mat.T @ w)

        # store estimates
        x_estimates[istart:istop] = X
        current_time = time_obs_iobs

    return x_estimates


@torch.no_grad()
def apply_EnKF(
    time_obs: _GenericTensor,
    gaps: _GenericTensor,
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
    r"""
    Implementation of the Ensemble Kalman Filter
    See e.g. Evensen, Ocean Dynamics (2003), Eqs. 44--54

    This function applies the Ensemble Kalman Filter algorithm to estimate
    the state of a dynamic system given noisy measurements.
    It uses an ensemble of state estimates to represent
    the uncertainty in the estimated state.
    It is executed within a no-grad context, meaning that gradient computation
    is disabled.

    Parameters
    ----------
    time_obs : _GenericTensor
        A 1D array containing the observation times in increasing order.

    gaps : _GenericTensor
        A 1D array containing the number of time steps
        between consecutive observations.

    Ne : int
        The number of ensemble members representing the state estimates.

    M : Callable
        The state transition function (process model) that predicts the state
        of the system given the previous state and the time range. It should
        have the signature
        M(x: torch.Tensor, time_range: torch.Tensor, \*args) -> torch.Tensor.
        'x' is the state vector, 'time_range' is a 1D tensor of time steps to
        predict the state forward, and '\*args' represents any additional
        arguments required by the state transition function.

    H : torch.Tensor | Callable
        The measurement matrix or a function that
        computes the measurement matrix. If 'H' is a torch.Tensor,
        it is a 2D tensor of shape (measurement_dim, state_dim),
        where 'measurement_dim' is the dimension of measurement
        and 'state_dim' is the dimension of the state vector.
        This matrix maps the state space to the measurement space.
        If 'H' is a Callable, it should have the signature
        H(x: torch.Tensor) -> torch.Tensor to compute the measurement,
        and 'H' must be able to handle the input 'x' with shape
        (number of ensemble, state_dim). The output of Callable 'H'
        must be a Tensor with shape (number of ensemble, measurement_dim).

    P0 : torch.Tensor
        The initial covariance matrix of the state estimate. A 2D tensor of
        shape (state_dim, state_dim). It represents the uncertainty of the
        initial state estimate.

    R : torch.Tensor
        The measurement noise covariance matrix. A 2D tensor of shape
        (measurement_dim, measurement_dim). It models the uncertainty in
        the measurements.

    x0 : torch.Tensor
        The initial state estimate. A 1D tensor of shape (state_dim).

    y : torch.Tensor
        The observed measurements. A 2D tensor of shape
        (number of observations, measurement_dim). Each row represents
        a measurement at a specific time step.

    start_time : float, optional
        The starting time of the filtering process. Default is 0.0.

    args : tuple, optional
        Additional arguments to pass to the state transition function 'M'.
        Default is (None,).

    Returns
    -------
    x_ave : torch.Tensor
        A 2D tensor of shape (num_steps + 1, state_dim). Each row represents
        the ensemble mean state vector at a specific time step, including
        the initial state.

    x_ens : torch.Tensor
        A 3D tensor of shape (num_steps + 1, Ne, state_dim). Each slice along
        the second dimension represents an ensemble member's state estimates
        at a specific time step, including the initial state.

    Raises
    ------
    TypeError
        If 'M' is not a callable or 'H' is not a torch.Tensor or Callable.

    Notes
    -----
    - The function assumes that the input tensors are properly shaped
      and valid for the Ensemble Kalman Filter. Ensure that 'x0', 'P0', 'R',
      and 'y' are appropriate for the dimensions of 'M' and 'H'.
    - The function assumes that 'time_obs' contains time points
      that are increasing, and 'gaps' specifies each number of time steps
      between consecutive observations.
    - The implementation uses an ensemble of state estimates to represent
      the uncertainty in the estimated state. The ensemble Kalman filter
      provides an approximation to the true state distribution.
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

    device = x0.device
    x_dim = x0.numel()
    num_steps = int(sum(gaps))
    x_ave = torch.zeros((num_steps + 1, x_dim), device=device)
    x_ens = torch.zeros((num_steps + 1, Ne, x_dim), device=device)

    # construct initial ensemble
    X = (
        torch.distributions.MultivariateNormal(
            loc=x0.ravel(), covariance_matrix=P0
        ).sample([Ne])
    ).to(device=device)

    # constants
    ONE_OVER_NE = 1.0 / Ne
    ONE_OVER_NE_MINUS_ONE = 1.0 / (Ne - 1.0)

    current_time = start_time
    for iobs, (time_obs_iobs, gap) in enumerate(zip(time_obs, gaps)):
        istart = iobs * gap
        istop = istart + gap + 1
        running_mean = torch.zeros((gap + 1, x_dim), device=device)
        time_fw = torch.linspace(
            current_time, time_obs_iobs, gap + 1, device=device
        )
        for e in range(Ne):
            # prediction phase for each ensemble member
            Xf = M(X[e], time_fw, *args)
            x_ens[istart:istop, e] = Xf
            X[e] = Xf[-1]
            running_mean = running_mean + Xf
        # updating phase
        # noise observations (Burgers et al, 1998)
        observations = (
            torch.distributions.MultivariateNormal(
                loc=y[iobs].ravel(),
                covariance_matrix=R,
            ).sample([Ne])
        ).to(device=device)
        X_mean = torch.mean(X, dim=0).view((1, -1))
        if isinstance(H, Callable):
            Xh = H(X)
            Xh_minus_z_mean = Xh - torch.mean(Xh, dim=0).view((1, -1))
            Pzz = (
                ONE_OVER_NE_MINUS_ONE * (Xh_minus_z_mean.T @ Xh_minus_z_mean)
                + R
            )
            Pxz = ONE_OVER_NE_MINUS_ONE * ((X - X_mean).T @ Xh_minus_z_mean)
            # Update
            X = X + (observations - Xh) @ torch.linalg.solve(Pzz, Pxz.T)
        else:  # isinstance(H, torch.Tensor)
            X_minus_X_mean = X - X_mean
            Pe = ONE_OVER_NE_MINUS_ONE * (X_minus_X_mean.T @ X_minus_X_mean)
            # Assembly of the Kalman gain matrix
            K = (H @ Pe @ H.T) + R
            # Solve
            w = torch.linalg.solve(K, observations.T - (H @ X.T))
            # Update
            X = X + (Pe @ H.T @ w).T
        running_mean = ONE_OVER_NE * running_mean
        x_ave[istart:istop] = running_mean
        current_time = time_obs_iobs

    return x_ave, x_ens
