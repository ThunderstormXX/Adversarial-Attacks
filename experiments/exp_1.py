import os
import sys
# _project_dir = os.path.dirname(os.getcwd())
_project_dir = os.getcwd()
os.environ['PROJECT_DIR'] = _project_dir
sys.path.append(_project_dir)
# print(os.getcwd())
print(_project_dir)
del _project_dir


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt

from src.utils import MNIST, train
from src.models import CNN
from src.advattack.attacks import test, plot_examples

from src.advattack.FGSM import FGSM
from src.advattack.noising import RandomTransform
