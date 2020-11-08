import glob
from torch.utils import data
import numpy as np
import torch
from myargs import args
from torchvision.transforms import Normalize
from itertools import chain
import tsaug
import random


data_mean = (0.5, )
data_std = (0.35,)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, impath, eval, pretraining):
        'Initialization'

        self.eval = eval
        self.augmenter = (
            #tsaug.TimeWarp() @ 0.5
            #tsaug.Drift(max_drift=(0.01, 0.1)) @ 0.5
            tsaug.Dropout(
                p=0.05,
                fill=0,
                size=[5, 10, 20]
            ) @ 0.5
        )

        # add to std if you would like some noise to be added to the spectrogram image
        self.std = 0

        # add images in different folders to the datalist
        datalist = []
        filepaths = glob.glob('{}/*'.format(impath))

        # open the gt.npy file in the folder and build a datalist
        if not pretraining:
            label_dict = np.load('./gt.npy', allow_pickle=True).flatten()[0]
        else:
            label_dict = np.load('./pretrain_gt.npy', allow_pickle=True).flatten()[0]

        for path in filepaths:
            datalist.append({
                'image': path,
                'label': label_dict[path.split('\\')[-1]]
            })

        self.datalist = datalist

        if not self.eval:
            self.datalist = list(chain(*[[i] * 1 for i in self.datalist]))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'
        image = np.load(self.datalist[index]['image'])
        label = self.datalist[index]['label']

        image = normalizepatch(image, self.eval, self.std, self.augmenter)

        return image, label


def GenerateIterator(args, impath, eval=False, shuffle=True, pretraining=False):
    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(impath, eval=eval, pretraining=pretraining), **params)


def normalizepatch(p, eval, std, augmenter):

    # put p between 0 and 1
    p = p - p.min()
    p = p / p.max()

    # we can consider adding this back in
    if not eval:

        # with 50% chance to do noise addition
        if random.random() > -1:
            noise = np.random.normal(0, std, (args.patch_classes, args.seq_length))
            p += noise

        # 50% for each of the separate augmenations in augmenter
        # for i in range(args.patch_classes):
        #     p[i, :] = augmenter.augment(p[i, :])

    p = np.ascontiguousarray(p)
    p = torch.from_numpy(p).unsqueeze(0).float()

    # consider normalizing here!
    p = Normalize(data_mean, data_std)(p)

    # output shape is (1, channels, sequence) typically (1, 8, 500)
    return p


