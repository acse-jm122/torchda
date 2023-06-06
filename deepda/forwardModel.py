import torch
from scipy.integrate import solve_ivp

__all__ = ["forwardModel_r"]


def Lorenz63(t, y, sig, rho, beta):
    # Lorenz '63 model
    y = torch.tensor(y)
    out = torch.zeros_like(y)
    out[0] = sig * (y[1] - y[0])
    out[1] = y[0] * (rho - y[2]) - y[1]
    out[2] = y[0] * y[1] - beta * y[2]
    return out


def forwardModel_r(xt0: torch.Tensor, time: torch.Tensor,
                   rayleigh, prandtl, b):
    #   perform integration of Lorentz63 model
    #   default integrator for solve_ivp is RK45

    # y0 = xt0.clone().detach()
    # y = torch.empty((xt0.size(0), time.size(0)), dtype=float)
    # y[:, 0] = y0
    # for i in range(1, int(time.size(0))):
    #     dy = Lorenz63(i, y0, prandtl, rayleigh, b)
    #     y0 += (time[i] - time[i - 1]) * dy
    #     y[:, i] = y0
    # return y
    rho = rayleigh
    beta = b
    sig = prandtl

    myParams = torch.tensor([sig, rho, beta], dtype=float)
    tstart = time[0]
    tend = time[-1]
    y0 = torch.tensor(xt0, dtype=float)
    sol = solve_ivp(Lorenz63, [tstart, tend], y0, args=myParams, dense_output=True)
    return torch.tensor(sol.sol(time))
