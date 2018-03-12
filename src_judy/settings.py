# @Author: Xi He <Heerye>
# @Date:   2018-03-01T11:53:04-05:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: settings.py
# @Last modified by:   Heerye
# @Last modified time: 2018-03-07T06:49:43-05:00


import argparse

import torch
torch_seed = 1

import numpy
numpy_seed = 17

parser = argparse.ArgumentParser(description='Eigenvalues')
parser.add_argument('-gpu', default=False,
                    type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-resume', default=False,
                    type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-reuse_params', default=False,
                    type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-n', default=300, type=int,
                    help='number of data points per cluster')
parser.add_argument('-i', default=2, type=int, help='index of data')
parser.add_argument('-b', default=50, type=int, help='batch size')
parser.add_argument('-N', default=50, type=int, help='maximal epochs')
parser.add_argument('-eps', default=1e-6, type=float, help='flatness threshold')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and args.gpu
data_index = args.i
n_points = args.n

train_ratio = 0.8
use_optimizer = False
advance_measure = False
