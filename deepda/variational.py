import torch
from typing import Callable

__all__ = ["apply_3DVar", "apply_4DVar"]


def apply_3DVar(
    H: Callable,
    B: torch.Tensor,
    R: torch.Tensor,
    xb: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 1e-5,
    max_iterations: int = 1000,
    learning_rate: float = 1e-3,
    logging: bool = True,
) -> torch.Tensor:
    """
    apply 3DVar
    """
    new_x0 = torch.nn.Parameter(xb.clone().detach())

    J_sum = lambda x: torch.sum(
        (x - xb).unsqueeze(1).T @ torch.linalg.solve(B, (x - xb))
        + (y - H(x)).unsqueeze(1).T @ torch.linalg.solve(R, y - H(x))
    )

    n = 0
    trainer = torch.optim.Adam([new_x0], lr=learning_rate)
    while True:
        trainer.zero_grad(set_to_none=True)
        loss = J_sum(new_x0)
        loss.backward()
        grad_norm = torch.norm(new_x0.grad)
        if logging:
            print(f"Iterations: {n}, Norm of J gradient: {grad_norm.item()}")
        if grad_norm <= threshold or n >= max_iterations:
            break
        trainer.step()
        n += 1

    return new_x0.detach()


def apply_4DVar(
    nobs: int,
    time_obs: torch.Tensor,
    gap: int,
    M: Callable,
    H: Callable,
    B: torch.Tensor,
    R: torch.Tensor,
    xb: torch.Tensor,
    y: torch.Tensor,
    model_args: tuple | None = None,
    threshold: float = 1e-5,
    max_iterations: int = 1000,
    learning_rate: float = 1e-3,
    logging: bool = True,
) -> torch.Tensor:
    """
    apply 4DVar
    """
    new_x0 = torch.nn.Parameter(xb.clone().detach())

    J_sum = lambda x, y: torch.sum(
        (x - xb).unsqueeze(1).T @ torch.linalg.solve(B, (x - xb))
        + (y - H(x)).unsqueeze(1).T @ torch.linalg.solve(R, y - H(x))
    )

    n = 0
    trainer = torch.optim.Adam([new_x0], lr=learning_rate)
    while True:
        trainer.zero_grad(set_to_none=True)
        current_time = 0
        total_loss = 0
        Xp = new_x0
        for iobs in range(nobs + 1):
            time_fw = torch.linspace(current_time, time_obs[iobs], gap + 1)
            xf = M(Xp, time_fw, *model_args)
            Xp = xf[:, -1]
            total_loss += J_sum(xf[:, -1], y[:, iobs])
            current_time = time_obs[iobs]
        total_loss.backward()
        grad_norm = torch.norm(new_x0.grad)
        if logging:
            print(f"Iterations: {n}, Norm of J gradient: {grad_norm.item()}")
        if grad_norm <= threshold or n >= max_iterations:
            break
        trainer.step()
        n += 1

    return new_x0.detach()
