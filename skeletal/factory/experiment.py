# Copyright (C) 2019 Ishaan Kumar
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>


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
