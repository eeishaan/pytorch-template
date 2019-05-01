import torch.nn as nn

class SimpleCriterFactory():
    @classmethod
    def make_criter(cls, criterion_name, params):
        return getattr(nn, criterion_name)(**params)
