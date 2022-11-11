from utility.data_preprocess.preprocess_util import *
from utility.data_loader.raw_loader import get_data, get_data_directories

# dirs = get_data_directories(False)
# data_preprocess(dirs)

import numpy as np
import torch

ddd = np.loadtxt('./data/e_nose_data_2022.dat')
device = torch.device('cuda')
print("CUDA available:", torch.cuda.is_available(), " GPU_name:", torch.cuda.get_device_name())

print("end")

