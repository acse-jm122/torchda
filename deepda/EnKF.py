import torch
from . import forwardModel_r

__all__ = ["myEnKF"]


def myEnKF(
    n_steps,
    dt,
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
):
    """
    Ensemble Kalman Filter
    """
    # Implementation of the ensemble Kalman filter
    #
    # See e.g. Evensen, Ocean Dynamics (2003), Eqs. 44--54
    #

    x = torch.zeros((3, nobs + 1), dtype=float)
    x_ave = torch.zeros((3, n_steps + 1), dtype=float)
    x_ens = torch.zeros((3, n_steps + 1, Ne), dtype=float)
    D = torch.zeros((H.size(0), Ne), dtype=float)
    Xe = torch.zeros((3, Ne), dtype=float)
    Xp = torch.zeros_like(Xe)
    sR = torch.sqrt(R)
    sP = torch.sqrt(P0)

    # construct initial ensemble
    Xe = x0.tile(Ne).reshape((-1, Ne)) + (
        sP @ torch.randn(size=(3, Ne), dtype=float))
    x[:, 0] = torch.mean(Xe, dim=1)
    one_over_Ne_minus_one = 1.0 / (Ne - 1.0)
    one_over_Ne = 1.0 / Ne

    current_time = 0
    for iobs in range(nobs + 1):
        istart = iobs * gap
        istop = istart + gap + 1
        running_mean = torch.zeros((3, gap + 1), dtype=float)
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
        K = (H @ (Pe @ H.T)) + R
        # Solve
        w = torch.linalg.solve(K, D - (H @ Xp))
        # Update
        Xe = Xp + (Pe @ (H.T @ w))
        running_mean = one_over_Ne * running_mean
        x_ave[:, istart:istop] = running_mean
        current_time = time_obs[iobs]

    return x_ave, x_ens
