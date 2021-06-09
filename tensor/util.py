import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import os
from plot import *
from conn import *
from glob import glob

def clean_data_dir(save_dir='data/'):

    files = glob(save_dir + '/*')
    for f in files:
        os.remove(f)

def stack_tensors(save_dir='data/'):

    files = glob(save_dir + '*.npz')
    files = sorted(glob(save_dir + '*.npz'))
    files = sorted([int(os.path.basename(file).split('.')[0]) for file in files])
    files = [save_dir+str(file)+'.npz' for file in files]

    #get the names of saved arrays
    keys = load_tensors(files[0]).keys()
    tensor_dict = {k: [] for k in keys}

    for file in files:
        dict = load_tensors(file)
        for key in dict.keys():
            tensor_dict[key].append(dict[key])
    for key in keys:
        tensor_dict[key] = np.stack(tensor_dict[key])

    return tensor_dict

def save_tensors(tensor_dict, tag, save_dir='data/', name=''):
    np.savez_compressed(save_dir + name + tag, **tensor_dict)

def load_tensors(path):
    data = np.load(path)
    tensor_dict = dict(data)
    data.close()
    return tensor_dict
