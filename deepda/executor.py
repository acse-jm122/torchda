from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from . import Algorithms, Device, _GenericTensor
from .kalman_filter import apply_EnKF
from .variational import apply_3DVar, apply_4DVar


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

    logging : bool, optional
        Whether to print log messages during execution. Default is True.

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
    logging: Optional[bool] = True
    args: Optional[tuple] = (None,)


class Executor:
    """
    Data assimilation executor class.

    This class provides a high-level integration for configuring and running
    data assimilation algorithms. It supports Ensemble Kalman Filter (EnKF),
    3D-Var, and 4D-Var.

    Methods
    -------
    set_input_parameters(parameters: Parameters) -> Executor:
        Set the input parameters for the data assimilation algorithm.

    run() -> dict[str, torch.Tensor | dict[str, list]]:
        Run the selected data assimilation algorithm and return results.

    get_results_dict() -> dict[str, torch.Tensor]:
        Get results dictionary.

    get_result(name: str) -> torch.Tensor:
        Get a specific result by name from the results dictionary.
    """

    def __init__(self) -> None:
        self.__parameters = None
        self.__results = {}

    def set_input_parameters(self, parameters: Parameters) -> "Executor":
        """
        Set the input parameters for the data assimilation algorithm.

        Parameters
        ----------
        parameters : Parameters
            An instance of the Parameters data class containing
            the configuration for the data assimilation algorithm.

        Returns
        -------
        Executor
            The Executor instance with updated input parameters.
        """
        self.__parameters = parameters
        return self

    def __check_EnKF_parameters(self) -> None:
        assert self.__parameters.num_steps > 0
        assert len(self.__parameters.observation_time_steps) >= 1
        assert self.__parameters.num_ensembles > 1

    def __check_3DVar_parameters(self) -> None:
        assert self.__parameters.max_iterations > 0
        assert self.__parameters.learning_rate > 0

    def __check_4DVar_parameters(self) -> None:
        self.__check_3DVar_parameters()
        assert isinstance(
            self.__parameters.observations, (tuple, list)
        ), "observations must be a tuple or list of Tensor in 4DVar"
        assert len(self.__parameters.observation_time_steps) >= 2

    def __call_apply_EnKF(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Call the apply_EnKF function with configured input parameters.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The results of the Ensemble Kalman Filter:
                average ensemble state estimates and
                individual ensemble state estimates.
        """
        return apply_EnKF(
            self.__parameters.num_steps,
            self.__parameters.observation_time_steps,
            self.__parameters.gap,
            self.__parameters.num_ensembles,
            self.__parameters.forward_model,
            self.__parameters.observation_model,
            self.__parameters.background_covariance_matrix,
            self.__parameters.observation_covariance_matrix,
            self.__parameters.background_state,
            self.__parameters.observations,
            self.__parameters.start_time,
            self.__parameters.args,
        )

    def __call_apply_3DVar(self) -> tuple[torch.Tensor, dict[str, list]]:
        """
        Call the apply_3DVar function with configured input parameters.

        Returns
        -------
        tuple[torch.Tensor, dict[str, list]]
            The results of the 3D-Var assimilation:
                assimilated background state estimate
                and intermediate optimization results.
        """
        return apply_3DVar(
            self.__parameters.observation_model,
            self.__parameters.background_covariance_matrix,
            self.__parameters.observation_covariance_matrix,
            self.__parameters.background_state,
            self.__parameters.observations,
            self.__parameters.max_iterations,
            self.__parameters.learning_rate,
            self.__parameters.logging,
        )

    def __call_apply_4DVar(self) -> tuple[torch.Tensor, dict[str, list]]:
        """
        Call the apply_4DVar function with configured input parameters.

        Returns
        -------
        tuple[torch.Tensor, dict[str, list]]
            The results of the 4D-Var assimilation:
                assimilated background state estimate
                and intermediate optimization results.
        """
        return apply_4DVar(
            self.__parameters.observation_time_steps,
            self.__parameters.gap,
            self.__parameters.forward_model,
            self.__parameters.observation_model,
            self.__parameters.background_covariance_matrix,
            self.__parameters.observation_covariance_matrix,
            self.__parameters.background_state,
            self.__parameters.observations,
            self.__parameters.max_iterations,
            self.__parameters.learning_rate,
            self.__parameters.logging,
            self.__parameters.args,
        )

    def __setup_device(self) -> None:
        """
        Set up the device for computation based on
        user-defined device preference.
        """
        device = (
            "cuda"
            if self.__parameters.device is Device.GPU
            and torch.cuda.is_available()
            else "cpu"
        )
        self.__parameters.background_covariance_matrix = (
            self.__parameters.background_covariance_matrix.to(device=device)
        )
        self.__parameters.observation_covariance_matrix = (
            self.__parameters.observation_covariance_matrix.to(device=device)
        )
        self.__parameters.background_state = (
            self.__parameters.background_state.to(device=device)
        )
        observations = self.__parameters.observations
        if isinstance(observations, (tuple, list)):
            for i, sample_iobs in enumerate(observations):
                observations[i] = sample_iobs.to(device=device)
        else:  # isinstance(observations, torch.Tensor)
            observations = observations.to(device=device)
        self.__parameters.observations = observations
        if (
            observation_model := self.__parameters.observation_model
        ) is not None and isinstance(observation_model, torch.Tensor):
            self.__parameters.observation_model = observation_model.to(
                device=device
            )

    def run(self) -> dict[str, torch.Tensor | dict[str, list]]:
        """
        Run the selected data assimilation algorithm and return results.

        Returns
        -------
        dict[str, torch.Tensor | dict[str, list]]
            A dictionary containing the results of the
            data assimilation algorithm.
        """
        self.__setup_device()
        if self.__parameters is None:
            raise RuntimeError("Set up input parameters before run.")
        algorithm = self.__parameters.algorithm
        if algorithm is Algorithms.EnKF:
            self.__check_EnKF_parameters()
            x_ave, x_ens = self.__call_apply_EnKF()
            self.__results = {
                "average_ensemble_all_states": x_ave,
                "each_ensemble_all_states": x_ens,
            }
        elif algorithm is Algorithms.Var3D:
            self.__check_3DVar_parameters()
            x0, intermediate_results = self.__call_apply_3DVar()
            self.__results = {
                "assimilated_background_state": x0,
                "intermediate_results": intermediate_results,
            }
        elif algorithm is Algorithms.Var4D:
            self.__check_4DVar_parameters()
            x0, intermediate_results = self.__call_apply_4DVar()
            self.__results = {
                "assimilated_background_state": x0,
                "intermediate_results": intermediate_results,
            }
        else:
            raise AttributeError(
                f"Unspported Algorithm: {algorithm} "
                "(Only support [EnKF, Var3D, Var4D])"
            )
        return deepcopy(self.__results)

    def get_results_dict(self) -> dict[str, torch.Tensor]:
        """
        Get a deep copy of the results dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            A deep copy of the results dictionary containing
            data assimilation results.
        """
        if self.__results:
            return deepcopy(self.__results)
        raise RuntimeError("No execution of the current case.")

    def get_result(self, name: str) -> torch.Tensor:
        """
        Get a deep copy of a specific result by name
        from the results dictionary.

        Parameters
        ----------
        name : str
            The name of the result to retrieve.

        Returns
        -------
        torch.Tensor
            A deep copy of the requested result.
        """
        if name in self.__results:
            return deepcopy(self.__results[name])
        raise KeyError(
            f"[{name}] is not a key in results dictionary. "
            f"Available keys: {self.__results.keys()}"
        )
