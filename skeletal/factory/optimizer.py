import torch.optim as optim


class SimpleOptimFactory:
    @classmethod
    def make_optim(cls, optimizer_name, params):
        return getattr(optim, optimizer_name)(**params)
