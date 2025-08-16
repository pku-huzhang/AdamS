import torch
from torch.optim.optimizer import Optimizer, required
from typing import Tuple


class AdamS(Optimizer):
    def __init__(self, optim_groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-6,
                 weight_decay=0.1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get raw gradient tensor
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamS does not support sparse gradients"
                    )

                lr = group['lr']
                eps = group['eps']
                wd = group['weight_decay']
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                # Identify if this is an embedding / norm parameter
                # Fetch or init state
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    
                state['step'] += 1

                bias_c1 = 1 - beta1 ** state['step']

                # weight decay
                p.data.mul_(1 - lr * wd)

                # Custom update for other weights

                m_hat = state['exp_avg'].div(bias_c1)

                # denominator
                variance = beta2 * m_hat * m_hat \
                            + (1 - beta2) * grad * grad
                denom = variance.sqrt_().add_(eps)
                
                state['exp_avg'].lerp_(grad, 1 - beta1)

                m_hat = state['exp_avg'].div(bias_c1)

                p.data.addcdiv_(m_hat, denom, value=-lr)

        return loss




class Lion(Optimizer):
    def __init__(
        self,
        optim_groups,
        lr: float = 3e-5,
        betas: Tuple[float, float] = (0.95, 0.98),
        weight_decay: float = 1.0,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])


        defaults = dict(
            lr = lr,
            betas = betas,
        )

        super().__init__(optim_groups, defaults)


    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                
                exp_avg = state['exp_avg']
                update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()

                # stepweight decay
                p.data.mul_(1 - lr * wd)

                # weight update
                p.add_(update, alpha = -lr)

                # decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

        return loss


