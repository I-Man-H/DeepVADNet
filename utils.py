import torch
import torch.nn as nn
import time
import os
from PIL import Image
from torch.utils import data
import numpy as np
import pandas as pd
from torchvision import transforms as T
import io
import zipfile
from torchnet import meter
from torch.utils.data import DataLoader
import scipy.io as scio



def out_put(string, verbose):
    '''
    Help function for verbose,
    output the string to destination path

    Parameters
    ----------
    string  :str,  the string to output
    verbose :str, the path to store the output
    '''
    with open(f"{verbose}.txt", "a") as f:
        f.write(string + "\n")
