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


import os
from pathlib import Path

# project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODEL_DIR = Path(os.path.abspath(PROJECT_ROOT / '../saved_models'))
RESULT_DIR = Path(os.path.abspath(PROJECT_ROOT / '../results'))
PARAM_DIR = PROJECT_ROOT / 'params'
LOG_DIR = PROJECT_ROOT / 'log'

# data file paths
DATA_ROOT_FOLDER = PROJECT_ROOT / 'data'
