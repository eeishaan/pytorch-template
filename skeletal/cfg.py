'''
Module containing singleton classes for configuration variables
'''

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
