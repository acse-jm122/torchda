from dataclasses import dataclass, field
from typing import Any, Callable, Type

import torch

from . import Algorithms, Device, _GenericTensor


@dataclass(slots=True)
class Parameters:
    """
    Data class to hold parameters for data assimilation.

    This class encapsulates the parameters required for data assimilation
    algorithms, such as Ensemble Kalman Filter (EnKF), 3D-Var, and 4D-Var.

    Attributes
    ----------
    algorithm : Algorithms
        The data assimilation algorithm to use (EnKF, 3D-Var, 4D-Var).

    device : Device, optional
        The device (CPU or GPU) to perform computations on. Default is CPU.

    observation_model : torch.Tensor | Callable[[torch.Tensor], torch.Tensor]
        The observation model or matrix 'H' that relates the state space to
        the observation space. It can be a pre-defined tensor or a Callable
        function that computes observations from the state.

    background_covariance_matrix : torch.Tensor
        The background covariance matrix 'B' representing the uncertainty
        of the background state estimate.

    observation_covariance_matrix : torch.Tensor
        The observation covariance matrix 'R' representing the uncertainty
        in the measurements.

    background_state : torch.Tensor
        The initial background state estimate 'xb'.

    observations : torch.Tensor
        The observed measurements corresponding to the given observation times.

    forward_model : Callable[[torch.Tensor, _GenericTensor], torch.Tensor] |
        Callable[..., torch.Tensor], optional
        The state transition function 'M' that predicts the state of the
        system given the previous state and the time range.
        Required for EnKF and 4D-Var.

    output_sequence_length : int, optional
        The number of output states along the time for the forward model.
        Used to determine the length of the output sequence
        for the forward model.

    observation_time_steps : _GenericTensor, optional
        A 1D array containing the observation times in increasing order.

    gaps : _GenericTensor, optional
        A 1D array containing the number of time steps
        between consecutive observations.

    num_ensembles : int, optional
        The number of ensembles used in the
        Ensemble Kalman Filter (EnKF) algorithm.

    start_time : int | float, optional
        The starting time of the data assimilation process. Default is 0.0.

    optimizer_cls : Type[torch.optim.Optimizer], optional
        The selected optimizer class for
        optimization-based algorithms (3D-Var, 4D-Var).

    optimizer_args : dict[str, Any], optional
        The hyperparameters in the selected optimizer class
        for optimization-based algorithms (3D-Var, 4D-Var).
        Default is {'lr': 1e-3}.

    max_iterations : int, optional
        The maximum number of iterations for
        optimization-based algorithms (3D-Var, 4D-Var).

    record_log : bool, optional
        Whether to record and print logs for iteration progress.
        Default is True.

    args : tuple, optional
        Additional arguments to pass to state transition function.

    Notes
    -----
    - Ensure that the provided tensors are properly shaped and compatible with
      the algorithm's requirements.
    - For EnKF, 'forward_model' should be provided,
      and 'observation_time_steps' should have at least 1 time point.
    - For 3D-Var and 4D-Var, 'optimizer_cls', 'optimizer_args',
      and 'max_iterations' control the optimization process.
    - For 4D-Var, 'observation_time_steps' should have at least 2 time points.
    """

    algorithm: Algorithms = None
    device: Device = Device.CPU
    observation_model: (
        torch.Tensor | Callable[[torch.Tensor], torch.Tensor]
    ) = None
    background_covariance_matrix: torch.Tensor = None
    observation_covariance_matrix: torch.Tensor = None
    background_state: torch.Tensor = None
    observations: torch.Tensor = None
    forward_model: (
        Callable[[torch.Tensor, _GenericTensor], torch.Tensor]
        | Callable[..., torch.Tensor]
    ) = lambda *args, **kwargs: None
    output_sequence_length: int = 1
    observation_time_steps: _GenericTensor = ()
    gaps: _GenericTensor = ()
    num_ensembles: int = 0
    start_time: int | float = 0.0
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    optimizer_args: dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3}
    )
    max_iterations: int = 1000
    record_log: bool = True
    args: tuple = ()
