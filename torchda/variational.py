from datetime import datetime
from typing import Any, Callable, Type

import torch

from . import _GenericTensor


def _J_dense(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return vector @ matrix @ vector


def _J_sparse(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return vector @ torch.sparse.mm(matrix, vector.view(-1, 1))


def _select_matrix_type(matrix: torch.Tensor) -> torch.Tensor:
    # if two thirds of elements in the matrix are zero,
    # then convert it to a sparse matrix.
    return (
        matrix.to_sparse()
        if (matrix.numel() / matrix.count_nonzero()) > 3
        else matrix
    )


def _select_compute_way(matrix: torch.Tensor) -> Callable:
    return _J_sparse if matrix.is_sparse else _J_dense


def _get_matrix_inv_and_J_func(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor,
           Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    matrix_inv = matrix.inverse()
    torch.cuda.empty_cache()
    matrix_inv = _select_matrix_type(matrix_inv)
    torch.cuda.empty_cache()
    J_func = _select_compute_way(matrix_inv)
    return matrix_inv, J_func


def apply_3DVar(
    H: Callable[[torch.Tensor], torch.Tensor],
    B: torch.Tensor,
    R: torch.Tensor,
    xb: torch.Tensor,
    y: torch.Tensor,
    optimizer_cls: Type[torch.optim.Optimizer] = None,
    optimizer_args: dict[str, Any] = None,
    max_iterations: int = 1000,
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
    H : Callable[[torch.Tensor], torch.Tensor]
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
        (state_dim,) or (batch_size, state_dim).

    y : torch.Tensor
        The observed measurements. A 1D or 2D tensor of shape
        (observation_dim,) or (batch_size, observation_dim).

    optimizer_cls : Type[torch.optim.Optimizer], optional
        The optimizer selected in the optimization algorithm.
        Default is 'AdamW'.

    optimizer_args : dict[str, Any], optional
        The hyperparameters in the selected optimizer class.
        Default is {'lr': 1e-3}.

    max_iterations : int, optional
        The maximum number of optimization iterations. Default is 1000.

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
        If 'H' is not a Callable.

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
            f"`H` must be a Callable in 3DVar, but given {type(H)=}"
        )

    xb_inner = xb.unsqueeze(0) if xb.ndim == 1 else xb
    y_inner = y.unsqueeze(0) if y.ndim == 1 else y

    new_x0 = torch.nn.Parameter(xb_inner.detach().clone())

    B_inv, Jb = _get_matrix_inv_and_J_func(B)
    R_inv, Jo = _get_matrix_inv_and_J_func(R)

    intermediate_results = {
        "J": [0] * max_iterations,
        "J_grad_norm": [0] * max_iterations,
        "background_states": [0] * max_iterations,
    }

    if optimizer_cls is None:
        optimizer_cls = torch.optim.AdamW
    if optimizer_args is None:
        optimizer_args = {"lr": 1e-3}

    optimizer = optimizer_cls([new_x0], **optimizer_args)
    batch_size = xb_inner.size(0)

    log_file = None
    try:
        log_template = (
            "Timestamp: {timestamp}, "
            "Iterations: {iteration}, J: {loss_J}, "
            "Norm of J gradient: {J_grad_norm}"
        )
        if record_log:
            # Set up logging with a timestamp in the log file name
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            log_file = open(f"3dvar_data_assimilation_{timestamp}.log", "w")

        for n in range(max_iterations):
            optimizer.zero_grad(set_to_none=True)
            loss_J = 0
            for i in range(batch_size):
                one_x0 = new_x0[i].ravel()
                x0_minus_xb = one_x0 - xb_inner[i].ravel()
                y_minus_H_x0 = (
                    y_inner[i].ravel() - H(one_x0.view(1, -1)).ravel()
                )
                loss_J += Jb(x0_minus_xb, B_inv) + Jo(y_minus_H_x0, R_inv)
            loss_J.backward(retain_graph=True)

            loss_J = loss_J.item()
            J_grad_norm = torch.norm(new_x0.grad).item()
            log_message = log_template.format(
                timestamp=datetime.now(),
                iteration=n,
                loss_J=loss_J,
                J_grad_norm=J_grad_norm,
            )
            print(log_message)
            if record_log:
                print(log_message, file=log_file)

            optimizer.step()
            intermediate_results["J"][n] = loss_J
            intermediate_results["J_grad_norm"][n] = J_grad_norm
            latest_x0 = new_x0.detach().clone().view_as(xb)
            intermediate_results["background_states"][n] = latest_x0
    finally:
        if log_file:
            log_file.close()

    return latest_x0, intermediate_results


def apply_4DVar(
    time_obs: _GenericTensor,
    gaps: _GenericTensor,
    M: (
        Callable[[torch.Tensor, _GenericTensor], torch.Tensor]
        | Callable[..., torch.Tensor]
    ),
    H: Callable[[torch.Tensor], torch.Tensor],
    B: torch.Tensor,
    R: torch.Tensor,
    xb: torch.Tensor,
    y: torch.Tensor,
    *args,
    optimizer_cls: Type[torch.optim.Optimizer] = None,
    optimizer_args: dict[str, Any] = None,
    max_iterations: int = 1000,
    record_log: bool = True,
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

    gaps : _GenericTensor
        A 1D array containing the number of time steps
        between consecutive observations.

    M : Callable[[torch.Tensor, _GenericTensor], torch.Tensor] |
        Callable[..., torch.Tensor]
        The state transition function (process model) that predicts the state
        of the system given the previous state and the time range.
        It should have the signature
        M(x: torch.Tensor, time_range: torch.Tensor, \*args) -> torch.Tensor.
        'x' is the state vector, 'time_range' is a 1D tensor of time steps to
        predict the state forward, and '\*args' represents any additional
        arguments required by the state transition function.

    H : Callable[[torch.Tensor], torch.Tensor]
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

    y : torch.Tensor
        The observed measurements. A 2D tensor of shape
        (number of observations, measurement_dim).
        Each row represents a measurement at a specific time step.

    args : tuple, optional
        Additional arguments to pass to the state transition function 'M'.

    optimizer_cls : Type[torch.optim.Optimizer], optional
        The optimizer class selected in the optimization algorithm.
        Default is 'AdamW'.

    optimizer_args : dict[str, Any], optional
        The hyperparameters in the selected optimizer class.
        Default is {'lr': 1e-3}.

    max_iterations : int, optional
        The maximum number of optimization iterations. Default is 1000.

    record_log : bool, optional
        Whether to record and print logs for iteration progress.
        Default is True.

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
        If 'M' or 'H' are not Callable, or if 'y' is not a tuple or list.

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
            f"`M` must be a Callable in 4DVar, but given {type(M)=}"
        )
    if not isinstance(H, Callable):
        raise TypeError(
            f"`H` must be a Callable in 4DVar, but given {type(H)=}"
        )

    new_x0 = torch.nn.Parameter(xb.detach().clone())

    B_inv, Jb = _get_matrix_inv_and_J_func(B)
    R_inv, Jo = _get_matrix_inv_and_J_func(R)

    intermediate_results = {
        "Jb": [0] * max_iterations,
        "Jo": [0] * max_iterations,
        "J": [0] * max_iterations,
        "J_grad_norm": [0] * max_iterations,
        "background_states": [0] * max_iterations,
    }

    if optimizer_cls is None:
        optimizer_cls = torch.optim.AdamW
    if optimizer_args is None:
        optimizer_args = {"lr": 1e-3}

    optimizer = optimizer_cls([new_x0], **optimizer_args)
    device = xb.device

    log_file = None
    try:
        log_template = (
            "Timestamp: {timestamp}, "
            "Iterations: {iteration}, Jb: {loss_Jb}, Jo: {loss_Jo}, "
            "J: {loss_J}, Norm of J gradient: {J_grad_norm}"
        )
        if record_log:
            # Set up logging with a timestamp in the log file name
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            log_file = open(f"4dvar_data_assimilation_{timestamp}.log", "w")

        for n in range(max_iterations):
            optimizer.zero_grad(set_to_none=True)
            current_time = time_obs[0]
            # loss_Jb = Jb(new_x0, xb, y)
            x0_minus_xb = new_x0.ravel() - xb.ravel()
            y_minus_H_x0 = y[0].ravel() - H(new_x0.view(1, -1)).ravel()
            loss_Jb = Jb(x0_minus_xb, B_inv)
            x = new_x0
            loss_Jo = Jo(y_minus_H_x0, R_inv)
            for iobs, (time_ibos, gap) in enumerate(
                zip(time_obs[1:], gaps), start=1
            ):
                time_fw = torch.linspace(
                    current_time, time_ibos, gap + 1, device=device
                )
                x = M(x, time_fw, *args)[-1]
                # loss_Jo += Jo(x, y[iobs])
                y_minus_H_xp = y[iobs].ravel() - H(x.view(1, -1)).ravel()
                loss_Jo += Jo(y_minus_H_xp, R_inv)
                current_time = time_ibos
            loss_J = loss_Jb + loss_Jo
            loss_J.backward(retain_graph=True)

            loss_Jb, loss_Jo = loss_Jb.item(), loss_Jo.item()
            loss_J = loss_J.item()
            J_grad_norm = torch.norm(new_x0.grad).item()
            log_message = log_template.format(
                timestamp=datetime.now(),
                iteration=n,
                loss_Jb=loss_Jb,
                loss_Jo=loss_Jo,
                loss_J=loss_J,
                J_grad_norm=J_grad_norm,
            )
            print(log_message)
            if record_log:
                print(log_message, file=log_file)

            optimizer.step()
            intermediate_results["Jb"][n] = loss_Jb
            intermediate_results["Jo"][n] = loss_Jo
            intermediate_results["J"][n] = loss_J
            intermediate_results["J_grad_norm"][n] = J_grad_norm
            latest_x0 = new_x0.detach().clone()
            intermediate_results["background_states"][n] = latest_x0
    finally:
        if log_file:
            log_file.close()

    return latest_x0, intermediate_results
