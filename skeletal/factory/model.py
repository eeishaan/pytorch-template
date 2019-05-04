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


'''
Supported model index
'''
from skeletal.model.cnn import CnnModel


class SimpleModelFactory(object):
    MODELS = {
        'cnn': CnnModel
    }

    @classmethod
    def make_model(cls, name, params):
        return cls.MODELS[name](**params)

    @classmethod
    def supported_models(cls):
        return cls.MODELS.keys()
