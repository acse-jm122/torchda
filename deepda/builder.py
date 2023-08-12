from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable

import torch
from numpy import ndarray
from numpy.linalg import LinAlgError

from . import Algorithms, Device, _GenericTensor
from .executor import _Executor
from .parameters import Parameters


class CaseBuilder:
    r"""
    A builder class for configuring and executing
    data assimilation cases.

    This class provides a convenient way for users to set up
    and execute data assimilation cases using various algorithms,
    including Ensemble Kalman Filter (EnKF), 3D-Var, and 4D-Var.

    Parameters
    ----------
    case_name : str, optional
        A name for the data assimilation case.
        Default is 'case_{current_timestamp}'.

    parameters : dict[str, Any] | Parameters, optional
        A dictionary or an instance of Parameters class containing
        configuration parameters for the data assimilation case.

    Attributes
    ----------
    case_name : str
        A name for the data assimilation case.

    Methods
    -------
    set_parameters(parameters: dict[str, Any] | Parameters)
        -> CaseBuilder:
        Set a batch of parameters for the data assimilation case.

    set_parameter(self, name: str, value: Any) -> CaseBuilder:
        Set a parameter for the data assimilation case.

    set_algorithm(algorithm: Algorithms) -> CaseBuilder:
        Set the data assimilation algorithm to use (EnKF, 3D-Var, 4D-Var).

    set_device(device: Device) -> CaseBuilder:
        Set the device (CPU or GPU) for computations.

    set_forward_model(forward_model: Callable) -> CaseBuilder:
        Set the state transition function 'M' for EnKF.

    set_output_sequence_length(output_sequence_length: int)-> "CaseBuilder":
        Set the output sequence length for the forward model.

    set_observation_model(observation_model: torch.Tensor | Callable)
        -> CaseBuilder:
        Set the observation model or matrix 'H' for EnKF.

    set_background_covariance_matrix
        (background_covariance_matrix: torch.Tensor) -> CaseBuilder:
        Set the background covariance matrix 'B'.

    set_observation_covariance_matrix
        (observation_covariance_matrix: torch.Tensor) -> CaseBuilder:
        Set the observation covariance matrix 'R'.

    set_background_state(background_state: torch.Tensor) -> CaseBuilder:
        Set the initial background state estimate 'xb'.

    set_observations(observations: torch.Tensor) -> CaseBuilder:
        Set the observed measurements.

    set_observation_time_steps(observation_time_steps: _GenericTensor)
        -> CaseBuilder:
        Set the observation time steps.

    set_gap(gap: int) -> CaseBuilder:
        Set the number of time steps between observations.

    set_num_steps(num_steps: int) -> CaseBuilder:
        Set the number of time steps to propagate the state forward.

    set_num_ensembles(num_ensembles: int) -> CaseBuilder:
        Set the number of ensembles for EnKF.

    set_start_time(start_time: int | float) -> CaseBuilder:
        Set the starting time of the data assimilation process.

    set_args(args: tuple) -> CaseBuilder:
        Set additional arguments for state transition function.

    set_max_iterations(max_iterations: int) -> CaseBuilder:
        Set the maximum number of iterations for
        optimization-based algorithms.

    set_learning_rate(learning_rate: int | float) -> CaseBuilder:
        Set the learning rate for optimization-based algorithms.

    set_record_log(record_log: bool) -> CaseBuilder:
        Set whether to record and print log messages during execution.

    execute() -> dict[str, torch.Tensor | dict[str, list]]:
        Execute the data assimilation case and return the results.

    get_results_dict() -> dict[str, torch.Tensor | dict[str, list]]:
        Get the dictionary containing the results of the executed case.

    get_parameters_dict() -> dict[str, Any]:
        Get the dictionary of configured parameters for the case.
    """

    def __init__(
        self,
        case_name: str = None,
        parameters: dict[str, Any] | Parameters = None,
    ) -> None:
        if case_name is None:
            self.case_name = datetime.now().strftime(
                "case_%Y-%m-%dT%H-%M-%S.%f"
            )
        self.__parameters = Parameters()
        if parameters is not None:
            self.set_parameters(parameters)
        self.__executor = _Executor()

    def set_parameters(
        self, parameters: dict[str, Any] | Parameters
    ) -> "CaseBuilder":
        if isinstance(parameters, Parameters):
            parameters = asdict(parameters)
        checked_builder = CaseBuilder()
        for param_name, param_value in parameters.items():
            checked_builder.set_parameter(param_name, param_value)
        self.__parameters = checked_builder.__parameters
        return self

    def set_parameter(self, name: str, value: Any) -> "CaseBuilder":
        if not hasattr(self.__parameters, name):
            raise AttributeError(
                f"Parameter '{name}' does not exist in Parameters."
            )
        setter_method = getattr(self, f"set_{name}", None)
        if setter_method is None:
            raise AttributeError(
                f"Setter method for parameter '{name}' not found."
            )
        return setter_method(value)

    def set_algorithm(self, algorithm: Algorithms) -> "CaseBuilder":
        if algorithm in Algorithms:
            self.__parameters.algorithm = algorithm
        return self

    def set_device(self, device: Device) -> "CaseBuilder":
        if device in Device:
            self.__parameters.device = device
        return self

    def set_forward_model(self, forward_model: Callable) -> "CaseBuilder":
        if not isinstance(forward_model, Callable):
            raise TypeError(
                "forward_model must be a callable type, "
                f"given {type(forward_model)=}"
            )
        self.__parameters.forward_model = forward_model
        return self

    def set_output_sequence_length(
        self, output_sequence_length: int
    ) -> "CaseBuilder":
        if not isinstance(output_sequence_length, Callable):
            raise TypeError(
                "output_sequence_length must be an integer, "
                f"given {type(output_sequence_length)=}"
            )
        self.__parameters.output_sequence_length = output_sequence_length
        return self

    def set_observation_model(
        self, observation_model: torch.Tensor | Callable
    ) -> "CaseBuilder":
        if not isinstance(observation_model, (torch.Tensor | Callable)):
            raise TypeError(
                "observation_model must be an instance of Tensor "
                f"or a callable type, given {type(observation_model)=}"
            )
        self.__parameters.observation_model = (
            observation_model.detach().clone()
            if isinstance(observation_model, torch.Tensor)
            else observation_model
        )
        return self

    @staticmethod
    def check_covariance_matrix(cov_matrix: torch.Tensor) -> None:
        """
        Check if a given covariance matrix is valid.

        Parameters
        ----------
        cov_matrix : torch.Tensor
            The covariance matrix to be checked.

        Raises
        ------
        LinAlgError
            If the covariance matrix is not a valid square matrix,
            not symmetric, or is singular.
        """
        if cov_matrix.ndim != 2 or (
            size := cov_matrix.size(0)
        ) != cov_matrix.size(1):
            raise LinAlgError(
                "Covariance matrix should be a 2D square matrix."
            )
        if not torch.allclose(cov_matrix, cov_matrix.T):
            raise LinAlgError(
                "Covariance matrix should be a symmetric matrix."
            )
        if torch.linalg.matrix_rank(cov_matrix) != size:
            raise LinAlgError("The input matrix is a singular matrix.")

    def set_background_covariance_matrix(
        self, background_covariance_matrix: torch.Tensor
    ) -> "CaseBuilder":
        if not isinstance(background_covariance_matrix, torch.Tensor):
            raise TypeError(
                "background_covariance_matrix must be an instance of Tensor, "
                f"given {type(background_covariance_matrix)=}"
            )
        self.check_covariance_matrix(background_covariance_matrix)
        self.__parameters.background_covariance_matrix = (
            background_covariance_matrix.detach().clone()
        )
        return self

    def set_observation_covariance_matrix(
        self, observation_covariance_matrix: torch.Tensor
    ) -> "CaseBuilder":
        if not isinstance(observation_covariance_matrix, torch.Tensor):
            raise TypeError(
                "observation_covariance_matrix "
                "must be an instance of Tensor, "
                f"given {type(observation_covariance_matrix)=}"
            )
        self.check_covariance_matrix(observation_covariance_matrix)
        self.__parameters.observation_covariance_matrix = (
            observation_covariance_matrix.detach().clone()
        )
        return self

    def set_background_state(
        self, background_state: torch.Tensor
    ) -> "CaseBuilder":
        if not isinstance(background_state, torch.Tensor):
            raise TypeError(
                "background_state must be an instance of Tensor, "
                f"given {type(background_state)=}"
            )
        self.__parameters.background_state = background_state.detach().clone()
        return self

    def set_observations(self, observations: torch.Tensor) -> "CaseBuilder":
        if not isinstance(observations, torch.Tensor):
            raise TypeError(
                "observations must be an instance of Tensor, "
                f"given {type(observations)=}"
            )
        self.__parameters.observations = observations.detach().clone()
        return self

    def set_observation_time_steps(
        self, observation_time_steps: _GenericTensor
    ) -> "CaseBuilder":
        if not isinstance(
            observation_time_steps, (list, tuple, ndarray, torch.Tensor)
        ):
            raise TypeError(
                "observation_time_steps must be a "
                f"{_GenericTensor.__bound__} type, "
                f"given {type(observation_time_steps)=}"
            )
        self.__parameters.observation_time_steps = (
            observation_time_steps.detach().clone()
            if isinstance(observation_time_steps, torch.Tensor)
            else observation_time_steps
        )
        return self

    def set_gap(self, gap: int) -> "CaseBuilder":
        if not isinstance(gap, int):
            raise TypeError(f"gap must be an integer, given {type(gap)=}")
        self.__parameters.gap = gap
        return self

    def set_num_steps(self, num_steps: int) -> "CaseBuilder":
        if not isinstance(num_steps, int):
            raise TypeError(
                f"num_steps must be an integer, given {type(num_steps)=}"
            )
        self.__parameters.num_steps = num_steps
        return self

    def set_num_ensembles(self, num_ensembles: int) -> "CaseBuilder":
        if not isinstance(num_ensembles, int):
            raise TypeError(
                "num_ensembles must be an integer, "
                f"given {type(num_ensembles)=}"
            )
        self.__parameters.num_ensembles = num_ensembles
        return self

    def set_start_time(self, start_time: int | float) -> "CaseBuilder":
        if not isinstance(start_time, (int, float)):
            raise TypeError(
                "start_time must be an integer or a floating point number, "
                f"given {type(start_time)=}"
            )
        self.__parameters.start_time = start_time
        return self

    def set_args(self, args: tuple) -> "CaseBuilder":
        if not isinstance(args, tuple):
            raise TypeError(f"args must be a tuple, given {type(args)=}")
        self.__parameters.args = args
        return self

    def set_max_iterations(self, max_iterations: int) -> "CaseBuilder":
        if not isinstance(max_iterations, int):
            raise TypeError(
                "max_iterations must be an integer, "
                f"given {type(max_iterations)=}"
            )
        self.__parameters.max_iterations = max_iterations
        return self

    def set_learning_rate(self, learning_rate: int | float) -> "CaseBuilder":
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                "learning_rate must be an integer or "
                f"a floating point number, given {type(learning_rate)=}"
            )
        self.__parameters.learning_rate = learning_rate
        return self

    def set_record_log(self, record_log: bool) -> "CaseBuilder":
        if not isinstance(record_log, bool):
            raise TypeError(
                f"record_log must be a bool, given {type(record_log)=}"
            )
        self.__parameters.record_log = record_log
        return self

    def execute(self) -> dict[str, torch.Tensor | dict[str, list]]:
        return self.__executor.set_input_parameters(self.__parameters).run()

    def get_results_dict(self) -> dict[str, torch.Tensor | dict[str, list]]:
        return self.__executor.get_results_dict()

    def get_result(self, name: str) -> torch.Tensor | dict[str, list]:
        r"""
        Get a specific result from the executed case by name.

        Parameters
        ----------
        name : str
            The name of the result to retrieve.

            - 'average_ensemble_all_states'
                Only available when algorithm is ``Algorithms.EnKF``.

            - 'each_ensemble_all_states'
                Only available when algorithm is ``Algorithms.EnKF``.

            - 'assimilated_background_state'
                Only available when algorithm is ``Algorithms.Var3D``
                or ``Algorithms.Var4D``.

            - 'intermediate_results'
                Only available when algorithm is ``Algorithms.Var3D``
                or ``Algorithms.Var4D``.

        Returns
        -------
        torch.Tensor | dict[str, list]
            A requested result.
        """
        return self.__executor.get_result(name)

    def get_parameters_dict(self) -> dict[str, Any]:
        return asdict(self.__parameters)

    def __repr__(self) -> str:
        """
        Generate a string representation of the
        configured parameters for the case.
        """
        params_dict = self.get_parameters_dict()
        str_list = [
            f"Parameters for Case: {self.case_name}",
            "--------------------------------------",
        ]
        str_list.extend(
            f"{param_name}:\n{param_value}\n"
            for param_name, param_value in params_dict.items()
        )
        return "\n".join(str_list)
