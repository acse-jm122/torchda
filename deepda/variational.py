import logging
from datetime import datetime
from typing import Callable

import torch

from . import _GenericTensor


def apply_3DVar(
    H: Callable,
    B: torch.Tensor,
    R: torch.Tensor,
    xb: torch.Tensor,
    y: torch.Tensor,
    max_iterations: int = 1000,
    learning_rate: float = 1e-3,
    record_log: bool = True,
) -> tuple[torch.Tensor, dict[str, list]]:
    r"""
    Implementation of the 3D-Var (Three-Dimensional Variational) assimilation.

    This function applies the 3D-Var assimilation algorithm to estimate
    the state of a dynamic system given noisy measurements. It aims to find
    the optimal state that minimizes the cost function combining background
    error and observation error.

    Parameters
    ----------
    H : Callable
        The observation operator that maps the state space to the observation
        space.
        It should have the signature H(x: torch.Tensor) -> torch.Tensor.

    B : torch.Tensor
        The background error covariance matrix.
        A 2D tensor of shape (state_dim, state_dim).
        It represents the uncertainty in the background.

    R : torch.Tensor
        The observation error covariance matrix.
        A 2D tensor of shape (observation_dim, observation_dim).
        It models the uncertainty in the measurements.

    xb : torch.Tensor
        The background state estimate. A 1D or 2D tensor of shape
        (state_dim,) or (sequence_length, state_dim).

    y : torch.Tensor
        The observed measurements. A 1D or 2D tensor of shape
        (observation_dim,) or (sequence_length, observation_dim).

    max_iterations : int, optional
        The maximum number of optimization iterations. Default is 1000.

    learning_rate : float, optional
        The learning rate for the optimization algorithm. Default is 1e-3.

    record_log : bool, optional
        Whether to record and print logs for iteration progress.
        Default is True.

    Returns
    -------
    x_optimal : torch.Tensor
        The optimal state estimate obtained using the 3D-Var assimilation.

    intermediate_results : dict[str, list]
        A dictionary containing intermediate results during optimization.

        - 'J'
            List of cost function values at each iteration.

        - 'J_grad_norm'
            List of norms of the cost function gradients at each iteration.

        - 'background_states'
            List of background state estimates at each iteration.

    Raises
    ------
    TypeError
        If 'H' is not a callable.

    Notes
    -----
    - The function assumes that the input tensors are properly shaped
      and valid for the 3D-Var assimilation. Ensure that 'xb', 'B', 'R',
      and 'y' are appropriate for the dimensions of 'H'.
    - The 3D-Var algorithm seeks an optimal state estimate by
      minimizing a cost function that incorporates both
      background and observation errors.
    """
    if not isinstance(H, Callable):
        raise TypeError(
            f"`H` must be a callable type in 3DVar, but given {type(H)=}"
        )
    if record_log:
        # Set up logging with a timestamp in the log file name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        logger = logging.getLogger(timestamp)
        logger.addHandler(
            logging.FileHandler(f"3dvar_data_assimilation_{timestamp}.log")
        )
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)

    xb_inner = xb.unsqueeze(0) if xb.ndim == 1 else xb
    y_inner = y.unsqueeze(0) if y.ndim == 1 else y

    new_x0 = torch.nn.Parameter(xb_inner.detach().clone())

    intermediate_results = {
        "J": [0] * max_iterations,
        "J_grad_norm": [0] * max_iterations,
        "background_states": [0] * max_iterations,
    }

    optimizer = torch.optim.Adam([new_x0], lr=learning_rate)
    sequence_length = xb_inner.size(0)
    for n in range(max_iterations):
        optimizer.zero_grad(set_to_none=True)
        loss_J = 0
        for i in range(sequence_length):
            one_x0 = new_x0[i].ravel()
            x0_minus_xb = one_x0 - xb_inner[i].ravel()
            y_minus_H_x0 = y_inner[i].ravel() - H(one_x0).ravel()
            loss_J += x0_minus_xb @ torch.linalg.solve(
                B, x0_minus_xb
            ) + y_minus_H_x0 @ torch.linalg.solve(R, y_minus_H_x0)
        loss_J.backward(retain_graph=True)
        loss_J, J_grad_norm = loss_J.item(), torch.norm(new_x0.grad).item()
        if record_log:
            logger.info(
                f"Iterations: {n}, J: {loss_J}, "
                f"Norm of J gradient: {J_grad_norm}"
            )
        optimizer.step()
        intermediate_results["J"][n] = loss_J
        intermediate_results["J_grad_norm"][n] = J_grad_norm
        latest_x0 = new_x0.detach().clone().view_as(xb)
        intermediate_results["background_states"][n] = latest_x0

    return latest_x0, intermediate_results


def apply_4DVar(
    time_obs: _GenericTensor,
    gap: int,
    M: Callable,
    H: Callable,
    B: torch.Tensor,
    R: torch.Tensor,
    xb: torch.Tensor,
    y: tuple[torch.Tensor] | list[torch.Tensor],
    max_iterations: int = 1000,
    learning_rate: float = 1e-3,
    record_log: bool = True,
    args: tuple = (None,),
) -> tuple[torch.Tensor, dict[str, list]]:
    r"""
    Implementation of the 4D-Var (Four-Dimensional Variational) assimilation.

    This function applies the 4D-Var assimilation algorithm to estimate
    the state of a dynamic system given noisy measurements. It aims to find
    the optimal state that minimizes the cost function combining background
    error and observation error over a specified time window.

    Parameters
    ----------
    time_obs : _GenericTensor
        A 1D array containing the observation times in increasing order.

    gap : int
        The number of time steps between consecutive observations.

    M : Callable
        The state transition function (process model) that predicts the state
        of the system given the previous state and the time range.
        It should have the signature
        M(x: torch.Tensor, time_range: torch.Tensor, \*args) -> torch.Tensor.
        'x' is the state vector, 'time_range' is a 1D tensor of time steps to
        predict the state forward, and '\*args' represents any additional
        arguments required by the state transition function.

    H : Callable
        The observation operator that maps the state space to the observation
        space.
        It should have the signature H(x: torch.Tensor) -> torch.Tensor.

    B : torch.Tensor
        The background error covariance matrix.
        A 2D tensor of shape (state_dim, state_dim).
        It represents the uncertainty in the background.

    R : torch.Tensor
        The observation error covariance matrix.
        A 2D tensor of shape (observation_dim, observation_dim).
        It models the uncertainty in the measurements.

    xb : torch.Tensor
        The background state estimate. A 1D tensor of shape (state_dim).

    y : tuple[torch.Tensor] | list[torch.Tensor]
        A tuple or list of observed measurements. Each element is a 1D tensor
        representing the measurements at a specific observation time.

    max_iterations : int, optional
        The maximum number of optimization iterations. Default is 1000.

    learning_rate : float, optional
        The learning rate for the optimization algorithm. Default is 1e-3.

    record_log : bool, optional
        Whether to record and print logs for iteration progress.
        Default is True.

    args : tuple, optional
        Additional arguments to pass to the state transition function 'M'.
        Default is (None,).

    Returns
    -------
    x_optimal : torch.Tensor
        The optimal state estimate obtained using the 4D-Var assimilation.

    intermediate_results : dict[str, list]
        A dictionary containing intermediate results during optimization.

        - 'Jb'
            List of background cost function values at each iteration.

        - 'Jo'
            List of observation cost function values at each iteration.

        - 'J'
            List of cost function values at each iteration.

        - 'J_grad_norm'
            List of norms of the cost function gradients at each iteration.

        - 'background_states'
            List of background state estimates at each iteration.

    Raises
    ------
    TypeError
        If 'M' or 'H' are not callable, or if 'y' is not a tuple or list.

    Notes
    -----
    - The function assumes that the input tensors are properly shaped
      and valid for the 4D-Var assimilation. Ensure that 'xb', 'B', 'R',
      and 'y' are appropriate for the dimensions of 'M', 'H',
      and the observation times.
    - The 4D-Var algorithm seeks an optimal state estimate over a time window
      by minimizing a cost function that incorporates both background and
      observation errors.
    """
    if not isinstance(M, Callable):
        raise TypeError(
            f"`M` must be a callable type in 4DVar, but given {type(M)=}"
        )
    if not isinstance(H, Callable):
        raise TypeError(
            f"`H` must be a callable type in 4DVar, but given {type(H)=}"
        )
    if not isinstance(y, (tuple, list)):
        raise TypeError(
            "`y` must be a tuple or list of Tensor in 4DVar, "
            f"but given {type(y)=}"
        )
    if record_log:
        # Set up logger with a timestamp in the log file name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        logger = logging.getLogger(timestamp)
        logger.addHandler(
            logging.FileHandler(f"4dvar_data_assimilation_{timestamp}.log")
        )
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)

    new_x0 = torch.nn.Parameter(xb.detach().clone())

    intermediate_results = {
        "Jb": [0] * max_iterations,
        "Jo": [0] * max_iterations,
        "J": [0] * max_iterations,
        "J_grad_norm": [0] * max_iterations,
        "background_states": [0] * max_iterations,
    }

    optimizer = torch.optim.Adam([new_x0], lr=learning_rate)
    device = xb.device
    for n in range(max_iterations):
        optimizer.zero_grad(set_to_none=True)
        current_time = time_obs[0]
        # loss_Jb = Jb(new_x0, xb, y)
        x0_minus_xb = new_x0.ravel() - xb.ravel()
        y_minus_H_x0 = y[0].ravel() - H(new_x0.ravel()).ravel()
        loss_Jb = x0_minus_xb @ torch.linalg.solve(
            B, x0_minus_xb
        ) + y_minus_H_x0 @ torch.linalg.solve(R, y_minus_H_x0)
        x = new_x0
        loss_Jo = 0
        for iobs, time_ibos in enumerate(time_obs[1:], start=1):
            time_fw = torch.linspace(
                current_time, time_ibos, gap + 1, device=device
            )
            x = M(x, time_fw, *args)
            for i in range(x.size(0)):  # sequence_length
                # loss_Jo += Jo(x[i], y[iobs][i])
                y_minus_H_xp = y[iobs][i].ravel() - H(x[i].ravel()).ravel()
                loss_Jo += y_minus_H_xp @ torch.linalg.solve(R, y_minus_H_xp)
            current_time = time_ibos
            x = x[-1]
        loss_J = loss_Jb + loss_Jo
        loss_J.backward(retain_graph=True)
        loss_Jb, loss_Jo = loss_Jb.item(), loss_Jo.item()
        loss_J, J_grad_norm = loss_J.item(), torch.norm(new_x0.grad).item()
        if record_log:
            logger.info(
                f"Iterations: {n}, Jb: {loss_Jb}, Jo: {loss_Jo}, "
                f"J: {loss_J}, Norm of J gradient: {J_grad_norm}"
            )
        optimizer.step()
        intermediate_results["Jb"][n] = loss_Jb
        intermediate_results["Jo"][n] = loss_Jo
        intermediate_results["J"][n] = loss_J
        intermediate_results["J_grad_norm"][n] = J_grad_norm
        latest_x0 = new_x0.detach().clone()
        intermediate_results["background_states"][n] = latest_x0

    return latest_x0, intermediate_results
