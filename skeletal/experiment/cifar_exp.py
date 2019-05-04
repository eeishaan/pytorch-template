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

from skeletal.cfg import DEVICE
from skeletal.experiment import BaseExperiment


class CifarExperiment(BaseExperiment):
    def before_train_forwardp(self, ctx, data):
        ctx.labels = data[1].to(DEVICE)
        return data[0]

    def compute_loss(self, ctx, outputs, labels):
        return self._criterion(outputs, ctx.labels)
