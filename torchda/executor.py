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

    Parameters
    ----------
    parameters : Parameters, optional
        An instance of Parameters class containing
        configuration parameters for the data assimilation case.
    """

    def __init__(self, parameters: Parameters = None) -> None:
        self.__parameters = parameters
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
        assert (
            self.__parameters.output_sequence_length > 0
        ), "`output_sequence_length` should be at least 1."
        assert (
            len_observation_time_steps := len(
                self.__parameters.observation_time_steps
            )
        ) >= 1, "`observation_time_steps` should be at least 1 in EnKF."
        assert len(self.__parameters.gaps) == len_observation_time_steps, (
            "The length of `gaps` should be equal to "
            "the length of `observation_time_steps`."
        )
        assert (
            self.__parameters.num_ensembles >= 2
        ), "`num_ensembles` should be at least 2 in EnKF."

    def __check_3DVar_parameters(self) -> None:
        assert (
            self.__parameters.max_iterations > 0
        ), "`max_iterations` should be greater than 0."
        assert issubclass(
            self.__parameters.optimizer_cls, torch.optim.Optimizer
        ), "`optimizer_cls` should be a subclass of `torch.optim.Optimizer`."

    def __check_4DVar_parameters(self) -> None:
        self.__check_3DVar_parameters()
        assert (
            self.__parameters.output_sequence_length > 0
        ), "`output_sequence_length` should be at least 1."
        assert (
            len_observation_time_steps := len(
                self.__parameters.observation_time_steps
            )
        ) >= 2, "`observation_time_steps` should be at least 2 in 4D-Var."
        assert len(self.__parameters.gaps) == (
            len_observation_time_steps - 1
        ), (
            "The length of `gaps` should be equal to "
            "the length of `observation_time_steps` minus 1."
        )

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
            self.__parameters.observation_time_steps,
            self.__parameters.gaps,
            self.__parameters.num_ensembles,
            self.__parameters.forward_model,
            self.__parameters.observation_model,
            self.__parameters.background_covariance_matrix,
            self.__parameters.observation_covariance_matrix,
            self.__parameters.background_state,
            self.__parameters.observations,
            *self.__parameters.args,
            start_time=self.__parameters.start_time,
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
            self.__parameters.optimizer_cls,
            self.__parameters.optimizer_args,
            self.__parameters.max_iterations,
            self.__parameters.record_log,
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
            self.__parameters.gaps,
            self.__parameters.forward_model,
            self.__parameters.observation_model,
            self.__parameters.background_covariance_matrix,
            self.__parameters.observation_covariance_matrix,
            self.__parameters.background_state,
            self.__parameters.observations,
            *self.__parameters.args,
            optimizer_cls=self.__parameters.optimizer_cls,
            optimizer_args=self.__parameters.optimizer_args,
            max_iterations=self.__parameters.max_iterations,
            record_log=self.__parameters.record_log,
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
        self.__parameters.observations = self.__parameters.observations.to(
            device=device
        )
        if isinstance(
            observation_model := self.__parameters.observation_model,
            (torch.Tensor, torch.nn.Module),
        ):
            self.__parameters.observation_model = observation_model.to(
                device=device
            )
        # `forward_model` is optional for 3D-Var algorithm. Default is `None`.
        if isinstance(
            forward_model := self.__parameters.forward_model, torch.nn.Module
        ):
            self.__parameters.forward_model = forward_model.to(device=device)

    def __wrap_forward_model(self) -> None:
        r"""
        Wrap the forward model with additional functionality if necessary.

        If the provided forward model is an instance of ``torch.nn.Module``,
        this method wraps it with a new function that handles the output
        sequence length for the state propagation.

        This is required to ensure compatibility with the data assimilation
        algorithms, particularly when the output sequence length is greater
        than 1.

        The wrapped forward model function is stored back to
        the ``forward_model`` attribute of the parameters object.
        """
        if isinstance(self.__parameters.forward_model, torch.nn.Module):
            forward_model = self.__parameters.forward_model

            def forward_model_wrapper(
                xb: torch.Tensor, time_fw: torch.Tensor, *args
            ):
                outs = [x0.view(1, -1) if (x0 := xb).ndim == 1 else xb]
                x = outs[0]
                forward_times, residue = divmod(
                    len(time_fw[:-1]),
                    self.__parameters.output_sequence_length
                )
                for _ in range(forward_times + 1):
                    x = forward_model(x, *args)
                    outs.append(x)
                    x = x[-1]
                outs[-1] = outs[-1][:residue]
                return torch.cat(outs)

            self.__parameters.forward_model = forward_model_wrapper

    def run(self) -> dict[str, torch.Tensor | dict[str, list]]:
        r"""
        Run the selected data assimilation algorithm and return results.

        Returns
        -------
        dict[str, torch.Tensor | dict[str, list]]
            A dictionary containing the results of the
            data assimilation algorithm.

            - 'average_ensemble_all_states'
                Only available when algorithm is ``Algorithms.EnKF``.

            - 'each_ensemble_all_states'
                Only available when algorithm is ``Algorithms.EnKF``.

            - 'assimilated_state'
                Only available when algorithm is ``Algorithms.Var3D``
                or ``Algorithms.Var4D``.

            - 'intermediate_results'
                Only available when algorithm is ``Algorithms.Var3D``
                or ``Algorithms.Var4D``.
        """
        self.__setup_device()
        self.__wrap_forward_model()
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
                "assimilated_state": x0,
                "intermediate_results": intermediate_results,
            }
        elif algorithm is Algorithms.Var4D:
            self.__check_4DVar_parameters()
            x0, intermediate_results = self.__call_apply_4DVar()
            self.__results = {
                "assimilated_state": x0,
                "intermediate_results": intermediate_results,
            }
        else:
            raise AttributeError(
                f"Unspported Algorithm: {algorithm} "
                "(Only support [EnKF, Var3D, Var4D])"
            )
        return deepcopy(self.__results)

    def get_results_dict(self) -> dict[str, torch.Tensor | dict[str, list]]:
        r"""
        Get a deep copy of the results dictionary.

        Returns
        -------
        dict[str, torch.Tensor | dict[str, list]]
            A deep copy of the results dictionary containing
            data assimilation results.

            - 'average_ensemble_all_states'
                Only available when algorithm is ``Algorithms.EnKF``.

            - 'each_ensemble_all_states'
                Only available when algorithm is ``Algorithms.EnKF``.

            - 'assimilated_state'
                Only available when algorithm is ``Algorithms.Var3D``
                or ``Algorithms.Var4D``.

            - 'intermediate_results'
                Only available when algorithm is ``Algorithms.Var3D``
                or ``Algorithms.Var4D``.
        """
        if self.__results:
            return deepcopy(self.__results)
        raise RuntimeError("No execution of the current case.")

    def get_result(self, name: str) -> torch.Tensor | dict[str, list]:
        r"""
        Get a deep copy of a specific result by name
        from the results dictionary.

        Parameters
        ----------
        name : str
            The name of the result to retrieve.

            - 'average_ensemble_all_states'
                Only available when algorithm is ``Algorithms.EnKF``.

            - 'each_ensemble_all_states'
                Only available when algorithm is ``Algorithms.EnKF``.

            - 'assimilated_state'
                Only available when algorithm is ``Algorithms.Var3D``
                or ``Algorithms.Var4D``.

            - 'intermediate_results'
                Only available when algorithm is ``Algorithms.Var3D``
                or ``Algorithms.Var4D``.

        Returns
        -------
        torch.Tensor | dict[str, list]
            A deep copy of the requested result.
        """
        if name in self.__results:
            return deepcopy(self.__results[name])
        raise KeyError(
            f"[{name}] is not a key in results dictionary. "
            f"Available keys: {self.__results.keys()}"
        )
