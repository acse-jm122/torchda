from copy import deepcopy

import torch

from . import Algorithms, Device
from .kalman_filter import apply_EnKF
from .parameters import Parameters
from .variational import apply_3DVar, apply_4DVar


class _Executor:
    """
    Data assimilation executor class.

    This class provides a high-level integration for configuring and running
    data assimilation algorithms. It supports Ensemble Kalman Filter (EnKF),
    3D-Var, and 4D-Var.

    Warning
    -------
    There is no validation for input parameters in this class.
    """

    def __init__(self) -> None:
        self.__parameters = None
        self.__results = {}

    def set_input_parameters(self, parameters: Parameters) -> "_Executor":
        """
        Set the input parameters for the data assimilation algorithm.

        Parameters
        ----------
        parameters : Parameters
            An instance of the Parameters data class containing
            the configuration for the data assimilation algorithm.

        Returns
        -------
        _Executor
            The _Executor instance with updated input parameters.
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
