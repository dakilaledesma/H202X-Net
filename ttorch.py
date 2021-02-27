# Import the core modules, check which GPU we end up with and scale batch size accordingly
import torch

torch.backends.cudnn.benchmark = True

import timm
from timm.data import *
from timm.utils import *

import pandas as pd
import numpy as np
import pynvml
from collections import OrderedDict
import logging
import time


def log_gpu_memory():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    info.free = round(info.free / 1024 ** 2)
    info.used = round(info.used / 1024 ** 2)
    logging.info('GPU memory free: {}, memory used: {}'.format(info.free, info.used))
    return info.used


def get_gpu_memory_total():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    info.total = round(info.total / 1024 ** 2)
    return info.total


setup_default_logging()

print('PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('CUDA available')
    device = 'cuda'
else:
    print('CUDA is not available')
    device = 'cpu'
