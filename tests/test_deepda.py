from pytest import fixture


@fixture(scope="module")
def torch():
    import torch

    return torch


@fixture(scope="module")
def torchda():
    import torchda

    return torchda


def test_import(torchda):
    assert torchda


@fixture(scope="module")
def dummy_tensor(torch):
    return torch.eye(3)


@fixture(scope="module")
def algorithms(torchda):
    return torchda.Algorithms


@fixture(scope="module")
def devices(torchda):
    return torchda.Device


@fixture(scope="module")
def parameters(torchda):
    return torchda.Parameters()


@fixture(scope="module")
def case(torchda):
    return torchda.CaseBuilder()


@fixture(scope="module")
def executor(torchda):
    return torchda._Executor()


def test_algorithms(algorithms):
    assert algorithms.EnKF in algorithms
    assert algorithms.Var3D in algorithms
    assert algorithms.Var4D in algorithms


def test_devices(devices):
    assert devices.CPU in devices
    assert devices.GPU in devices


def test_parameters_slots(parameters):
    try:
        parameters.a_nonexistent_attr = "something"
        assert False, (
            "An instance of Parameters class should not"
            " have undefined parameters."
        )
    except AttributeError:
        assert True


def test_case_repr(case):
    assert repr(case)
    assert str(case)


def test_parameters_attributes(parameters):
    for key in (
        "algorithm",
        "device",
        "observation_model",
        "background_covariance_matrix",
        "observation_covariance_matrix",
        "background_state",
        "observations",
        "forward_model",
        "output_sequence_length",
        "observation_time_steps",
        "gaps",
        "num_ensembles",
        "start_time",
        "optimizer_cls",
        "optimizer_args",
        "max_iterations",
        "early_stop",
        "record_log",
        "args",
    ):
        assert hasattr(parameters, key)


def test_case_attributes(case):
    assert hasattr(case, "case_name")


def test_case_set_parameter(case):
    try:
        case.set_parameter("nonexistent_attr", None)
        assert False, "No setter for a nonexistent attribute."
    except AttributeError:
        assert True


def test_case_set_algorithm(case, algorithms):
    assert case.set_algorithm(algorithms.EnKF)
    assert case.set_algorithm(algorithms.Var3D)
    assert case.set_algorithm(algorithms.Var4D)
    assert case.set_parameter("algorithm", algorithms.EnKF)
    assert case.set_parameter("algorithm", algorithms.Var3D)
    assert case.set_parameter("algorithm", algorithms.Var4D)


def test_case_set_device(case, devices):
    assert case.set_device(devices.CPU)
    assert case.set_device(devices.GPU)
    assert case.set_parameter("device", devices.CPU)
    assert case.set_parameter("device", devices.GPU)


def test_case_set_forward_model(case):
    assert case.set_forward_model(lambda _: None)
    assert case.set_parameter("forward_model", lambda _: None)
    try:
        case.set_parameter("forward_model", 0)
        assert False, "forward_model must be a callable."
    except TypeError:
        assert True


def test_case_set_output_sequence_length(case):
    assert case.set_output_sequence_length(1)
    assert case.set_parameter("output_sequence_length", 1)
    try:
        case.set_parameter("output_sequence_length", "1")
        assert False, "output_sequence_length must be an integer."
    except TypeError:
        assert True


def test_case_set_observation_model(case, dummy_tensor):
    assert case.set_observation_model(lambda _: None)
    assert case.set_observation_model(dummy_tensor)
    assert case.set_parameter("observation_model", lambda _: None)
    assert case.set_parameter("observation_model", dummy_tensor)
    try:
        case.set_parameter("observation_model", 0)
        assert (
            False
        ), "observation_model must be a callable or an instance of Tensor."
    except TypeError:
        assert True


def test_case_check_covariance_matrix(torchda, case, dummy_tensor):
    try:
        case.check_covariance_matrix(dummy_tensor)
        torchda.CaseBuilder.check_covariance_matrix(dummy_tensor)
    except Exception:
        assert False


def test_case_set_background_covariance_matrix(case, dummy_tensor):
    assert case.set_background_covariance_matrix(dummy_tensor)
    assert case.set_parameter("background_covariance_matrix", dummy_tensor)
    try:
        case.set_parameter("background_covariance_matrix", lambda _: None)
        assert (
            False
        ), "background_covariance_matrix must be an instance of Tensor."
    except TypeError:
        assert True


def test_case_set_observation_covariance_matrix(case, dummy_tensor):
    assert case.set_observation_covariance_matrix(dummy_tensor)
    assert case.set_parameter("observation_covariance_matrix", dummy_tensor)
    try:
        case.set_parameter("observation_covariance_matrix", lambda _: None)
        assert (
            False
        ), "observation_covariance_matrix must be an instance of Tensor."
    except TypeError:
        assert True


def test_case_set_background_state(case, dummy_tensor):
    assert case.set_background_state(dummy_tensor)
    assert case.set_parameter("background_state", dummy_tensor)
    try:
        case.set_parameter("background_state", lambda _: None)
        assert False, "background_state must be an instance of Tensor."
    except TypeError:
        assert True


def test_case_set_observations(case, dummy_tensor):
    assert case.set_observations(dummy_tensor)
    assert case.set_parameter("observations", dummy_tensor)
    try:
        case.set_parameter("observations", lambda _: None)
        assert False, "observations must be an instance of Tensor."
    except TypeError:
        assert True


def test_case_set_observation_time_steps(case, dummy_tensor):
    assert case.set_observation_time_steps([])
    assert case.set_observation_time_steps(())
    assert case.set_observation_time_steps(dummy_tensor.ravel())
    assert case.set_observation_time_steps(dummy_tensor.ravel().numpy())
    assert case.set_parameter("observation_time_steps", [])
    assert case.set_parameter("observation_time_steps", ())
    assert case.set_parameter("observation_time_steps", dummy_tensor.ravel())
    assert case.set_parameter(
        "observation_time_steps", dummy_tensor.ravel().numpy()
    )
    try:
        case.set_parameter("observation_time_steps", "0")
        assert (
            False
        ), "observation_time_steps must be a list | tuple | ndarray | Tensor."
    except TypeError:
        assert True


def test_case_set_gaps(case, dummy_tensor):
    assert case.set_gaps([])
    assert case.set_gaps(())
    assert case.set_gaps(dummy_tensor.ravel())
    assert case.set_gaps(dummy_tensor.ravel().numpy())
    assert case.set_parameter("gaps", [])
    assert case.set_parameter("gaps", ())
    assert case.set_parameter("gaps", dummy_tensor.ravel())
    assert case.set_parameter("gaps", dummy_tensor.ravel().numpy())
    try:
        case.set_parameter("gaps", "0")
        assert False, "gaps must be a list | tuple | ndarray | Tensor."
    except TypeError:
        assert True


def test_case_set_num_ensembles(case):
    assert case.set_num_ensembles(2)
    assert case.set_parameter("num_ensembles", 2)
    try:
        case.set_parameter("num_ensembles", "2")
        assert False, "num_ensembles must be an integer."
    except TypeError:
        assert True


def test_case_set_start_time(case):
    assert case.set_start_time(0)
    assert case.set_start_time(10.5)
    assert case.set_parameter("start_time", 0)
    assert case.set_parameter("start_time", 11.2)
    try:
        case.set_parameter("start_time", "0.25")
        assert (
            False
        ), "start_time must be an integer or a floating point number."
    except TypeError:
        assert True


def test_case_set_args(case):
    assert case.set_args(())
    assert case.set_parameter("args", ())
    try:
        case.set_parameter("args", "0")
        assert False, "args must be a tuple."
    except TypeError:
        assert True


def test_case_set_max_iterations(case):
    assert case.set_max_iterations(1000)
    assert case.set_parameter("max_iterations", 1)
    try:
        case.set_parameter("max_iterations", "1")
        assert False, "max_iterations must be an integer."
    except TypeError:
        assert True


def test_case_set_optimizer_cls(case, torch):
    assert case.set_optimizer_cls(torch.optim.Adam)
    assert case.set_parameter("optimizer_cls", torch.optim.Adam)
    try:
        case.set_parameter("optimizer_cls", torch.Tensor)
        assert (
            False
        ), "optimizer_cls must be a subclass of torch.optim.Optimizer."
    except TypeError:
        assert True


def test_case_set_optimizer_args(case):
    assert case.set_optimizer_args({"lr": 1e-3})
    assert case.set_parameter("optimizer_args", {"lr": 1e-3})
    try:
        case.set_parameter("optimizer_args", [])
        assert False, "optimizer_args must be an instance of dict[str, Any]."
    except TypeError:
        assert True


def test_early_stop(case):
    assert case.set_early_stop((50, 0.01))
    assert case.set_early_stop((0, 0))
    assert case.set_parameter("early_stop", None)

    def assert_raise(input):
        try:
            case.set_parameter("early_stop", input)
            assert (
                False
            ), "early_stop must be a (int, int | float) tuple or None."
        except TypeError:
            assert True

    assert_raise("0")
    assert_raise([5, 0.0])
    assert_raise((0, 0, 0))
    assert_raise((1.25, 0))
    assert_raise((1, "0"))


def test_case_set_record_log(case):
    assert case.set_record_log(True)
    assert case.set_record_log(False)
    assert case.set_parameter("record_log", True)
    assert case.set_parameter("record_log", False)
    try:
        case.set_parameter("record_log", None)
        assert False, "record_log must be a bool."
    except TypeError:
        assert True


def test_case_get_parameters_dict(case):
    assert case.get_parameters_dict()


def test_executor_get_results_dict(executor):
    try:
        executor.get_results_dict()
        assert (
            False
        ), "No attributes for the results dict from a not executed executor."
    except RuntimeError:
        assert True


def test_executor_get_result(executor):
    try:
        executor.get_result("nonexistent_attr")
        assert False, "`nonexistent_attr` is not a key in results dictionary."
    except KeyError:
        assert True
