import glob
from torch.utils import data
import numpy as np
import torch
from myargs import args
from torchvision.transforms import Normalize
from itertools import chain
import tsaug
import random
from scipy.signal import spectrogram


data_mean = (0.5, )
data_std = (0.35,)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, impath, eval, spectro_in):
        'Initialization'

        self.eval = eval
        self.spectro_in = spectro_in
        self.augmenter = (
            # tsaug.TimeWarp() @ 0.5 +
            tsaug.Drift(max_drift=(0.01, 0.1)) @ 0.5
            # tsaug.Dropout(
            #     p=0.05,
            #     fill=0,
            #     size=[5, 10, 20]
            # ) @ 0.5
        )

        # add to std if you would like some noise to be added to the spectrogram image
        self.std = 0

        # add images in different folders to the datalist
        datalist = []
        filepaths = glob.glob('{}/*'.format(impath))

        for path in filepaths:
            datadict = np.load(path, allow_pickle=True).flatten()[0]
            datalist.append(datadict)

        self.datalist = datalist

        if not self.eval:
            self.datalist = list(chain(*[[i] * 1 for i in self.datalist]))

        print(f'{"train" if not self.eval else "val"} dataset length: {len(datalist)}')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.datalist[index]['data']

        # convert from micro to millivolts?
        data = data / 1000

        # get label, convert to binary value
        label = self.datalist[index]['label']
        label = 0 if label == 'left' else 1

        data = normalizepatch(data, self.eval, self.std, self.augmenter, self.spectro_in)

        return data, label


def GenerateIterator(impath, eval=False, shuffle=True, spectro_in=False):
    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(impath, eval=eval, spectro_in=spectro_in), **params)


def normalizepatch(p, eval, std, augmenter, use_spectro):

    if use_spectro:
        # # ts aug before spectrogram
        # if not eval:
        #     # 50% for each of the separate augmenations in augmenter
        #     for i in range(len(p)):
        #         p[i, :] = augmenter.augment(p[i, :])

        # try spectrogram stuff again
        total_spectrograms = []
        for cha in range(len(p)):
            # use n per seg of 1024 to have a shape of 513x1
            f, t, spectro = spectrogram(p[cha, :], fs=512, nperseg=1024)

            # divide by mean of each channel to normalize, take only relevant frequencies
            spectro = spectro[14:62, 0].flatten()
            spectro = spectro / spectro.mean()

            total_spectrograms.extend(spectro)

        total_spectrograms = np.asarray(total_spectrograms)

        # put spec between -1 and 1
        # total_spectrograms = total_spectrograms - total_spectrograms.min()
        # total_spectrograms = total_spectrograms / total_spectrograms.max() * 2 - 1

        # # augmentations
        if not eval:

            # with 50% chance to do noise addition
            if random.random() > 0.5:
                noise = np.random.normal(0, std, len(p) * (62-14))
                total_spectrograms += noise

        # p = np.ascontiguousarray(total_spectrograms)
        p = torch.from_numpy(total_spectrograms).float()

    else:

        # put p between 0 and 1
        # p = p - p.min()
        # p = p / p.max()

        # augmentations
        if not eval:

            # with 50% chance to do noise addition
            if random.random() > -1:
                noise = np.random.normal(0, std, (len(p), args.seq_length))
                p += noise

            # 50% for each of the separate augmenations in augmenter
            # for i in range(len(p)):
            #     p[i, :] = augmenter.augment(p[i, :])

        p = np.ascontiguousarray(p)
        p = torch.from_numpy(p).float().unsqueeze(0)

    # consider normalizing here!
    # p = Normalize(data_mean, data_std)(p)

    # output shape is (1, channels, sequence) typically (1, 64, 1024)
    # or (64, 24, 24) for spectrogram
    # or (3072,) for only frequency information
    return p


# g = GenerateIterator('../data/preproc_online/', spectro_in=True)
# for x, y in g:
#     print(x.shape, y.shape)
#     break
