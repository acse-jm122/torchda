import torch
from . import forwardModel_r

__all__ = ["KF", "EnKF"]


def KF(
    n_steps,
    nobs,
    time_obs,
    gap,
    H,
    R,
    y,
    x0: torch.Tensor,
    P0,
    rayleigh,
    prandtl,
    b,
) -> torch.Tensor:
    """
    Implementation of the Kalman filter
    """
    x_estimates = torch.zeros((x0.size(0), n_steps + 1), dtype=float)
    sR = torch.sqrt(R)
    sP = torch.sqrt(P0)

    # construct initial state
    Xp = x0 + (sP @ torch.randn(size=(x0.size(0),), dtype=float))

    current_time = 0
    for iobs in range(nobs + 1):
        istart = iobs * gap
        istop = istart + gap + 1
        time_fw = torch.linspace(current_time, time_obs[iobs], gap + 1)

        # Time update (prediction)
        xf = forwardModel_r(Xp, time_fw, rayleigh, prandtl, b)
        Xp = xf[:, -1]

        # Noise the obs (Burgers et al, 1998)
        D = y[:, iobs] + (sR @ torch.randn(size=(H.size(0),)))
        P = torch.outer(Xp, Xp)
        K = (H @ P @ H.T) + R
        w = torch.linalg.solve(K, D - (H @ Xp))
        Xp = Xp + (P @ H.T @ w)

        # Store estimate
        x_estimates[:, istart:istop] = xf
        current_time = time_obs[iobs]
    return x_estimates


def EnKF(
    n_steps,
    nobs,
    time_obs,
    gap,
    Ne,
    H,
    R,
    y,
    x0: torch.Tensor,
    P0,
    rayleigh,
    prandtl,
    b,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of the Ensemble Kalman Filter
    See e.g. Evensen, Ocean Dynamics (2003), Eqs. 44--54
    """
    x_ave = torch.zeros((x0.size(0), n_steps + 1), dtype=float)
    x_ens = torch.zeros((x0.size(0), n_steps + 1, Ne), dtype=float)
    D = torch.zeros((H.size(0), Ne), dtype=float)
    Xp = torch.zeros((x0.size(0), Ne), dtype=float)
    sR = torch.sqrt(R)
    sP = torch.sqrt(P0)

    # construct initial ensemble
    Xe = x0.tile(Ne).reshape((-1, Ne)) + (
        sP @ torch.randn(size=(x0.size(0), Ne), dtype=float))
    one_over_Ne_minus_one = 1.0 / (Ne - 1.0)
    one_over_Ne = 1.0 / Ne

    current_time = 0
    running_mean = torch.empty((3, gap + 1), dtype=float)
    for iobs in range(nobs + 1):
        istart = iobs * gap
        istop = istart + gap + 1
        running_mean.zero_()
        time_fw = torch.linspace(current_time, time_obs[iobs], gap + 1)
        for e in range(Ne):
            # prediction phase for each ensemble member
            xf = forwardModel_r(Xe[:, e], time_fw, rayleigh, prandtl, b)
            x_ens[:, istart:istop, e] = xf
            Xp[:, e] = xf[:, -1]
            running_mean = running_mean + xf
            # Noise the obs (Burgers et al, 1998)
            D[:, e] = y[:, iobs] + (sR @ torch.randn(size=(H.size(0),)))
        E = torch.mean(Xp, dim=1)
        A = Xp - E.tile(Ne).reshape((-1, Ne))
        Pe = one_over_Ne_minus_one * (A @ A.T)
        # Assembly of the Kalman gain matrix
        K = (H @ Pe @ H.T) + R
        # Solve
        w = torch.linalg.solve(K, D - (H @ Xp))
        # Update
        Xe = Xp + (Pe @ H.T @ w)
        running_mean = one_over_Ne * running_mean
        x_ave[:, istart:istop] = running_mean
        current_time = time_obs[iobs]

    return x_ave, x_ens
