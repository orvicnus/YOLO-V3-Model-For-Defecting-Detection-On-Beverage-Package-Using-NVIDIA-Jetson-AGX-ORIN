# Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
import os
import torch
import pandas as pd
import numpy as np
import PIL

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch import nn
# Note: PyTorch 1.10.0+ is required for this course
torch.__version__
