from dataclasses import dataclass
from typing import Callable, Optional

import torch

from . import Algorithms, Device, _GenericTensor


@dataclass
class Parameters:
    """
    Data class to hold parameters for data assimilation.

    This class encapsulates the parameters required for data assimilation
    algorithms, such as Ensemble Kalman Filter (EnKF), 3D-Var, and 4D-Var.

    Attributes
    ----------
    algorithm : Algorithms
        The data assimilation algorithm to use (EnKF, 3D-Var, 4D-Var).

    observation_model : torch.Tensor | Callable, optional
        The observation model or matrix 'H' that relates the state space to
        the observation space. It can be a pre-defined tensor or a callable
        function that computes observations from the state.

    background_covariance_matrix : torch.Tensor, optional
        The background covariance matrix 'B' representing the uncertainty
        of the background state estimate.

    observation_covariance_matrix : torch.Tensor, optional
        The observation covariance matrix 'R' representing the uncertainty
        in the measurements.

    background_state : torch.Tensor, optional
        The initial background state estimate 'xb'.

    observations :
        torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor], optional
        The observed measurements corresponding to the given observation times.
        It can be a single tensor or a collection of tensors.

    device : Device, optional
        The device (CPU or GPU) to perform computations on. Default is CPU.

    forward_model : Callable, optional
        The state transition function 'M' that predicts the state of the
        system given the previous state and the time range.
        Required for EnKF and 4D-Var.

    observation_time_steps : _GenericTensor, optional
        A 1D array containing the observation times in increasing order.

    gap : int, optional
        The number of time steps between consecutive observations.

    num_steps : int, optional
        The number of time steps to propagate the state forward.
        Required for EnKF and 4D-Var.

    num_ensembles : int, optional
        The number of ensembles used in the
        Ensemble Kalman Filter (EnKF) algorithm.

    start_time : int | float, optional
        The starting time of the data assimilation process. Default is 0.0.

    max_iterations : int, optional
        The maximum number of iterations for
        optimization-based algorithms (3D-Var, 4D-Var).

    learning_rate : int | float, optional
        The learning rate for optimization-based algorithms (3D-Var, 4D-Var).

    record_log : bool, optional
        Whether to record and print logs for iteration progress.
        Default is True.

    args : tuple, optional
        Additional arguments to pass to state transition function.

    Notes
    -----
    - Ensure that the provided tensors are properly shaped and compatible with
      the algorithm's requirements.
    - For EnKF, `forward_model` should be provided,
      and `num_steps` should be > 0.
    - For 3D-Var and 4D-Var, `max_iterations` and `learning_rate` control the
      optimization process.
    - For 4D-Var, `observations` should be a tuple or list of tensors, and
      `observation_time_steps` should have at least 2 time points.
    """

    algorithm: Algorithms = Algorithms.EnKF
    observation_model: torch.Tensor | Callable = None
    background_covariance_matrix: torch.Tensor = None
    observation_covariance_matrix: torch.Tensor = None
    background_state: torch.Tensor = None
    observations: torch.Tensor | tuple[torch.Tensor] | list[
        torch.Tensor
    ] = None
    device: Optional[Device] = Device.CPU
    forward_model: Optional[Callable] = None
    observation_time_steps: Optional[_GenericTensor] = None
    gap: Optional[int] = 0
    num_steps: Optional[int] = 0
    num_ensembles: Optional[int] = 0
    start_time: Optional[int | float] = 0.0
    max_iterations: Optional[int] = 1000
    learning_rate: Optional[int | float] = 0.001
    record_log: Optional[bool] = True
    args: Optional[tuple] = (None,)
