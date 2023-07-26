from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import torch
from numpy import ndarray

from . import Algorithms
from .kalman_filter import apply_EnKF
from .variational import apply_3DVar, apply_4DVar

_Tensor = TypeVar(
    "_Tensor",
    bound=list | tuple | ndarray | torch.Tensor,
)


@dataclass
class Parameters:
    algorithm: Algorithms = Algorithms.EnKF
    observation_model: torch.Tensor | Callable = torch.nn.Module
    background_covariance_matrix: torch.Tensor = torch.Tensor
    observation_covariance_matrix: torch.Tensor = torch.Tensor
    background_state: torch.Tensor = torch.Tensor
    initial_state: torch.Tensor = torch.Tensor
    observations: torch.Tensor = torch.Tensor
    forward_model: Optional[Callable] = torch.nn.Module
    observation_time_steps: Optional[_Tensor] = torch.Tensor
    num_steps: Optional[int] = 0
    num_ensembles: Optional[int] = 0
    start_time: Optional[int | float] = 0.0
    args: Optional[tuple] = (None,)
    threshold: Optional[int | float] = 0.00001
    max_iterations: Optional[int] = 1000
    learning_rate: Optional[int | float] = 0.001
    is_vector_xb: Optional[bool] = True
    is_vector_y: Optional[bool] = True
    batch_first: Optional[bool] = True
    logging: Optional[bool] = True


class Executor:
    def __init__(self) -> None:
        self.__parameters = None
        self.__results = {}

    def set_input_parameters(self, parameters: Parameters) -> "Executor":
        self.__parameters = parameters

    def __check_EnKF_parameters(self) -> None:
        assert self.__parameters.num_steps > 0
        assert len(self.__parameters.observation_time_steps) >= 2
        assert self.__parameters.num_ensembles > 1

    def __check_3DVar_parameters(self) -> None:
        assert self.__parameters.threshold >= 0
        assert self.__parameters.max_iterations > 0
        assert self.__parameters.learning_rate > 0

    def __check_4DVar_parameters(self) -> None:
        self.__check_3DVar_parameters()
        assert len(self.__parameters.observation_time_steps) >= 2

    def __call_apply_EnKF(self) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_EnKF(
            self.__parameters.num_steps,
            len(self.__parameters.observation_time_steps),
            self.__parameters.observation_time_steps,
            (
                self.__parameters.observation_time_steps[1]
                - self.__parameters.observation_time_steps[0]
            ),
            self.__parameters.num_ensembles,
            self.__parameters.forward_model,
            self.__parameters.observation_model,
            self.__parameters.background_covariance_matrix,
            self.__parameters.observation_covariance_matrix,
            self.__parameters.initial_state,
            self.__parameters.observations,
            self.__parameters.start_time,
            self.__parameters.args,
        )

    def __call_apply_3DVar(self) -> torch.Tensor:
        return apply_3DVar(
            self.__parameters.observation_model,
            self.__parameters.background_covariance_matrix,
            self.__parameters.observation_covariance_matrix,
            self.__parameters.initial_state,
            self.__parameters.observations,
            self.__parameters.threshold,
            self.__parameters.max_iterations,
            self.__parameters.learning_rate,
            self.__parameters.is_vector_xb,
            self.__parameters.batch_first,
            self.__parameters.logging,
        )

    def __call_apply_4DVar(self) -> torch.Tensor:
        return apply_4DVar(
            len(self.__parameters.observation_time_steps),
            self.__parameters.observation_time_steps,
            (
                self.__parameters.observation_time_steps[1]
                - self.__parameters.observation_time_steps[0]
            ),
            self.__parameters.forward_model,
            self.__parameters.observation_model,
            self.__parameters.background_covariance_matrix,
            self.__parameters.observation_covariance_matrix,
            self.__parameters.initial_state,
            self.__parameters.observations,
            self.__parameters.start_time,
            self.__parameters.threshold,
            self.__parameters.max_iterations,
            self.__parameters.learning_rate,
            self.__parameters.is_vector_xb,
            self.__parameters.is_vector_y,
            self.__parameters.batch_first,
            self.__parameters.logging,
            self.__parameters.args,
        )

    def run(self) -> dict[str, torch.Tensor]:
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
            x0 = self.__call_apply_3DVar()
            self.__results = {"initial_state": x0}
        elif algorithm is Algorithms.Var4D:
            self.__check_4DVar_parameters()
            x0 = self.__call_apply_4DVar()
            self.__results = {"initial_state": x0}
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
            return self.__results[name].clone()
        raise KeyError(
            f"[{name}] is not a key in results dictionary. "
            f"Available keys: {self.__results.keys()}"
        )
