Use Deep Learning in Data Assimilation
======================================

deepda
------


Important Notes:

- Background state `x0` or `xb` must be a 1D or 2D tensor with shape 
  ([`batch size`], state dimension), and `batch size` is optional. 
  `batch size` is only available for 3D Variational (3D-Var) algorithm.

- Observations or measurements `y` must be a 2D tensor.

  - The shape of `y` must be (number of observations, state dimension) for Kalman Filter (KF), 
    Ensemble Kalman Filter (EnKF), and 4D Variational (4D-Var) algorithm. The number of 
    observations must be at least 1 in KF or EnKF, and this number must be at least 2 in 4D-Var.
  - The shape of `y` must be ([`batch size`], state dimension) for 3D-Var algorithm, and 
    `batch size` is optional for ``batch size == 1``.

- A callable code object `M` must be able to handle the 1D tensor input `x` with shape (state dimension,). 
  The output of the `M` must be a 2D tensor with shape (time window sequence length, state dimension).

- `H` could be either a tensor or a callable code object, and tensor `H` is only available in KF algorithm.

  - If `H` is a callable code object, it must be able to handle the input `x` in 2D tensor with 
    shape ([`number of ensemble`], state dimension), and default ``number of ensemble == 1`` 
    for all algorithms except EnKF. The output of this callable code object `H` must 
    be a corresponding shape ([`number of ensemble`], measurement dimension), and the output must be a 
    2D tensor with shape (1, state dimension) in all other algorithms.
  - If `H` is a tensor, it must be a 2D tensor of shape (measurement dimension, state dimension). 
    This matrix maps the state space to the measurement space.
  - Notice: A callable `H` code object would be approximated by the Jacobian method in KF algorithm. 
    That might cause unexpected error due to incorrect shaping of tensor. 
    Recommended input tensor `x` for the callable `H` is a 1D tensor (state dimension,) in KF algorithm.


.. automodule:: deepda
  :members: Parameters, CaseBuilder, apply_KF, apply_EnKF, apply_3DVar, apply_4DVar, _Executor
