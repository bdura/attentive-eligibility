import torch
from torch.optim.optimizer import Optimizer, required


class EligibilitySGD(Optimizer):
    def __init__(self, params, lr=required, gamma=required, lambd=required):
        defaults = dict(lr=lr, gamma=gamma, lambd=lambd)
        self.traces = {}
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

        for p in group['params']:
            e = self.traces.get(p, torch.zeros_like(p))
            if p.grad is None:
                continue
            d_p = p.grad.data
            decay = torch.mul(gamma, lambd)
            # update trace
            self.traces[p] = torch.add(torch.mul(decay, e), d_p / (-2 * d_t))
            factor = torch.mul(group['lr'], d_t)
            # update param
            p.data = torch.mul(self.traces[p], factor)

        return loss
