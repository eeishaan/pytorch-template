from skeletal.experiment import BaseExperiment


class SimpleExperimentFactory(object):
    SUPPORTED_EXP = {
        'base': BaseExperiment,
    }

    @classmethod
    def supported_experiments(cls):
        return cls.SUPPORTED_EXP.keys()

    @classmethod
    def make_experiment(cls, exp_name, params):
        """
        Embedding name has one-to-one map with experiment name
        """
        return cls.SUPPORTED_EXP[exp_name](**params)
