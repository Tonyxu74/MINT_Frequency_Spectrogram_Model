import glob
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import random
import os, os.path
import torch
from myargs import args

# findFile is actually unecessary at the moment
def findFile(root_dir, endswith):
    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(endswith):
                all_files.append(os.path.join(path, file))

    return all_files


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, impath, eval):
        'Initialization'

        self.eval = eval
        self.std = 0

        # add images in different folders to the datalist
        #datalist = []
        #self.trainfiles = glob.glob('{}/*'.format(impath))
        #self.length = 0 
        self.number_files = len(os.listdir('data/train/trainfiles')) 
        #get number of files and therefore events
        
        self.labelfile = np.load('data/train/gt.npy',allow_pickle=True)

        #for imfolder in trainfiles:
            # open the gt.npy file in the folder and build a datalist
            #print('placeholder')
        #self.datalist = [item for sublist in datalist for item in sublist]

        #if not self.eval:
         #   from itertools import chain
         #   self.datalist = list(chain(*[[i] * 1 for i in self.datalist]))

    def __len__(self):
        'Denotes the total number of samples'
        return self.number_files

    def __getitem__(self, index):
        'Generates one sample of data'

        data = np.load('data/train/trainfiles/{}.npy'.format(index),allow_pickle=True)
        label = self.labelfile.item()['{}.npy'.format(index)]
        #labelarray = np.zeros(args.classes)
        #labelarray[label]=1

        return data, label


def GenerateIterator(args, impath, eval=False, shuffle=True):
    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        #'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(impath, eval=eval), **params)


def normalizepatch(p, gt, eval, std):

    if not eval:
        rot_num = random.choice([0, 1, 2, 3])
        p = np.rot90(p, rot_num)
        gt = np.rot90(gt, rot_num)

        noise = np.random.normal(
            0, std, args.imageDims)
        p += noise

    p = np.ascontiguousarray(p)
    gt = np.ascontiguousarray(gt)


