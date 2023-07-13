import torch
from typing import Callable

__all__ = ["KF", "EnKF"]


def KF(
    n_steps: int,
    nobs: int,
    time_obs: torch.Tensor,
    gap: int,
    M: torch.Tensor | Callable,
    H: torch.Tensor,
    R: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor,
    P0: torch.Tensor,
    start_time: float = 0.0,
    args: tuple = (None,),
) -> torch.Tensor:
    """
    Implementation of the Kalman Filter (constant P assumption)
    """
    device = x0.device
    x_estimates = torch.zeros((x0.size(0), n_steps + 1), device=device)
    sP = P0.sqrt()
    P = P0

    # construct initial state
    Xp = x0 + (sP @ torch.randn(size=(x0.size(0),), device=device))

    current_time = start_time
    for iobs in range(nobs):
        istart = iobs * gap
        istop = istart + gap + 1

        # Time update (prediction)
        if isinstance(M, torch.Tensor):
            xf = M @ Xp
        elif isinstance(M, Callable):
            time_fw = torch.linspace(
                current_time, time_obs[iobs], gap + 1, device=device
            )
            xf = M(Xp, time_fw, *args)
        else:
            raise TypeError(
                f"Only support types: [Callable, torch.Tensor], \
                but given {type(M)=}"
            )
        Xp = xf[:, -1]

        if isinstance(M, torch.Tensor):
            P = M @ P @ M.T
            K = P @ H.T @ torch.linalg.pinv((H @ P @ H.T) + R)
            Xp = Xp + (K @ (y[:, iobs] - (H @ Xp)))
            P = (torch.eye(x0.size(0), device=device) - (K @ H)) @ P
        else:  # isinstance(M, Callable)
            K = (H @ P @ H.T) + R
            w = torch.linalg.solve(K, y[:, iobs] - (H @ Xp))
            Xp = Xp + (P @ H.T @ w)

        # Store estimate
        x_estimates[:, istart:istop] = xf
        current_time = time_obs[iobs]

    return x_estimates


def EnKF(
    n_steps: int,
    nobs: int,
    time_obs: torch.Tensor,
    gap: int,
    Ne: int,
    M: Callable,
    H: Callable,
    R: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor,
    P0: torch.Tensor,
    inflation_factor: float = 0.0,
    localization: torch.Tensor = None,
    start_time: float = 0.0,
    args: tuple = (None,),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of the Ensemble Kalman Filter
    See e.g. Evensen, Ocean Dynamics (2003), Eqs. 44--54
    """
    device = x0.device
    x_ave = torch.zeros((x0.size(0), n_steps + 1), device=device)
    x_ens = torch.zeros((x0.size(0), n_steps + 1, Ne), device=device)
    D = torch.zeros((y.size(0), Ne), device=device)
    Xp = torch.zeros((x0.size(0), Ne), device=device)
    sR = R.sqrt()
    sP = P0.sqrt()

    # construct initial ensemble
    Xe = x0.reshape((-1, 1)) + (
        sP @ torch.randn(size=(x0.size(0), Ne), device=device)
    )
    one_over_Ne_minus_one = 1.0 / (Ne - 1.0)
    one_over_Ne = 1.0 / Ne
    one_plus_inflation_factor = 1.0 + inflation_factor

    current_time = start_time
    running_mean = torch.empty((x0.size(0), gap + 1), device=device)
    for iobs in range(nobs):
        istart = iobs * gap
        istop = istart + gap + 1
        running_mean.zero_()
        time_fw = torch.linspace(
            current_time, time_obs[iobs], gap + 1, device=device
        )
        for e in range(Ne):
            # prediction phase for each ensemble member
            xf = M(Xe[:, e], time_fw, *args)
            x_ens[:, istart:istop, e] = xf
            Xp[:, e] = xf[:, -1]
            running_mean = running_mean + xf
            # Noise the obs (Burgers et al, 1998)
            D[:, e] = y[:, iobs] + (
                sR @ torch.randn(size=(y.size(0),), device=device)
            )
        E = torch.mean(Xp, dim=1).reshape((-1, 1))
        A = Xp - E
        Pe = one_plus_inflation_factor * one_over_Ne_minus_one * (A @ A.T)
        if localization is not None:
            Pe = localization * Pe
        if isinstance(H, Callable):
            Xh = H(Xp)
            z_mean = torch.mean(Xh, dim=1).reshape((-1, 1))
            Xh_minus_z_mean = Xh - z_mean
            Pzz = (
                one_over_Ne_minus_one * (Xh_minus_z_mean @ Xh_minus_z_mean.T)
                + R
            )
            Pxz = one_over_Ne_minus_one * ((Xp - E) @ Xh_minus_z_mean.T)
            Xe = Xp + Pxz @ torch.linalg.solve(Pzz, D - Xh)
        elif isinstance(H, torch.Tensor):
            # Assembly of the Kalman gain matrix
            K = (H @ Pe @ H.T) + R
            # Solve
            w = torch.linalg.solve(K, D - (H @ Xp))
            # Update
            Xe = Xp + (Pe @ H.T @ w)
        else:
            raise TypeError(
                f"Only support types: [Callable, torch.Tensor], \
                    but given {type(H)=}"
            )
        running_mean = one_over_Ne * running_mean
        x_ave[:, istart:istop] = running_mean
        current_time = time_obs[iobs]

    return x_ave, x_ens
