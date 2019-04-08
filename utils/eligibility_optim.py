import torch
from torch.optim.optimizer import Optimizer, required


class EligibilitySGD(Optimizer):
    def __init__(self, params, lr=required, gamma=required, lambd=required):
        defaults = dict(lr=lr, gamma=gamma, lambd=lambd)
        self.traces = (torch.zeros_like(p) for p in params)
        super(EligibilitySGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        pass

    def step(self, loss):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        d_t = torch.sqrt(loss)

        for group in self.param_groups:
            gamma = group['gamma']
            lambd = group['lambd']

        for p, e in zip(group['params'], self.traces):
            if p.grad is None:
                continue
            d_p = p.grad.data
            decay = torch.mul(gamma, lambd)
            # update trace
            e.add_(torch.mul(decay, e), d_p)

            # update param
            p.data.add_(e, torch.mul(-group['lr'], d_t))

        return loss
