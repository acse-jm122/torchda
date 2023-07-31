from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import torch
from numpy import ndarray

from . import Algorithms, Device
from .kalman_filter import apply_EnKF
from .variational import apply_3DVar, apply_4DVar

_GenericTensor = TypeVar(
    "_GenericTensor",
    bound=list | tuple | ndarray | torch.Tensor,
)


@dataclass
class Parameters:
    algorithm: Algorithms = Algorithms.EnKF
    observation_model: torch.Tensor | Callable = None
    background_covariance_matrix: torch.Tensor = None
    observation_covariance_matrix: torch.Tensor = None
    background_state: torch.Tensor = None
    observations: torch.Tensor = None
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
    def __init__(self) -> None:
        self.__parameters = None
        self.__results = {}

    def set_input_parameters(self, parameters: Parameters) -> "Executor":
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
        assert len(self.__parameters.observation_time_steps) >= 2

    def __call_apply_EnKF(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.__parameters.observations = self.__parameters.observations.to(
            device=device
        )
        if (
            observation_model := self.__parameters.observation_model
        ) is not None and isinstance(observation_model, torch.Tensor):
            self.__parameters.observation_model = observation_model.to(
                device=device
            )

    def run(self) -> dict[str, torch.Tensor | dict[str, list]]:
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
        if self.__results:
            return deepcopy(self.__results)
        raise RuntimeError("No execution of the current case.")

    def get_result(self, name: str) -> torch.Tensor:
        if name in self.__results:
            return deepcopy(self.__results[name])
        raise KeyError(
            f"[{name}] is not a key in results dictionary. "
            f"Available keys: {self.__results.keys()}"
        )
