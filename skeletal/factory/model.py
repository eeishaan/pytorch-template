'''
Supported model index
'''
from torchvision.models import resnet18



class SimpleModelFactory(object):
    MODELS = {
        'resnet': resnet18,
    }

    @classmethod
    def make_model(cls, name, params):
        return cls.MODELS[name](**params)

    @classmethod
    def supported_models(cls):
        return cls.MODELS.keys()

