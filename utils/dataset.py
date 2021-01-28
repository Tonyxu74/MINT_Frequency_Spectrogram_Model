import glob
from torch.utils import data
import numpy as np
import torch
from myargs import args
from torchvision.transforms import Normalize
from itertools import chain
import tsaug
import random
from scipy.signal import butter, lfilter, spectrogram


data_mean = (0.5, )
data_std = (0.35,)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, impath, eval, input_mode, spectro_in):
        'Initialization'

        self.eval = eval
        self.input_mode = input_mode
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

        # open the gt.npy file in the folder and build a datalist
        if input_mode == 'train':
            label_dict = np.load('./gt.npy', allow_pickle=True).flatten()[0]
        elif input_mode == 'pretrain':
            label_dict = np.load('./pretrain_gt.npy', allow_pickle=True).flatten()[0]
        elif input_mode == 'htrain':
            label_dict = np.load('./h_gt.npy', allow_pickle=True).flatten()[0]
        else:
            raise ValueError('input mode not supported')

        for path in filepaths:
            datalist.append({
                'image': path,
                'label': label_dict[path.split('\\')[-1]]
            })

        self.datalist = datalist
        print(f'{"train" if not self.eval else "val"} dataset length: {len(datalist)}')

        if not self.eval:
            self.datalist = list(chain(*[[i] * 10 for i in self.datalist]))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'
        image = np.load(self.datalist[index]['image'])

        # pretrined in millivolts, convert from micro to millivolts
        # if not self.pretraining:
        #     image = image / 1000

        label = self.datalist[index]['label']

        image = normalizepatch(image, self.eval, self.std, self.augmenter, self.spectro_in)

        return image, label


def GenerateIterator(args, impath, eval=False, shuffle=True, input_mode='train', spectro_in=False):
    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(impath, eval=eval, input_mode=input_mode, spectro_in=spectro_in), **params)


def butter_bandpass(lowcut, highcut, fs=250, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, low, btype='highpass', output='ba')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, order=5):
    b, a = butter_bandpass(lowcut, highcut, order=order)
    y = lfilter(b, a, data)
    return y


def normalizepatch(p, eval, std, augmenter, use_spectro):

    # filter here
    # for i in range(args.patch_classes):
    #     p[i] = butter_bandpass_filter(p[i], 1, 124, order=5)

    if use_spectro:
        # # ts aug before spectrogram
        # if not eval:
        #     # 50% for each of the separate augmenations in augmenter
        #     for i in range(args.patch_classes):
        #         p[i, :] = augmenter.augment(p[i, :])

        # try spectrogram stuff again
        total_spectrograms = []
        for cha in range(args.patch_classes):
            # use n per seg of 22 to have a shape of 12x12
            f, t, spectro = spectrogram(p[cha, :], fs=250, nperseg=22)
            total_spectrograms.append(spectro)
        total_spectrograms = np.asarray(total_spectrograms)

        # put spec between 0 and 1
        total_spectrograms = total_spectrograms - total_spectrograms.min()
        total_spectrograms = total_spectrograms / total_spectrograms.max()

        # augmentations
        if not eval:

            # with 50% chance to do noise addition
            if random.random() > 0.5:
                noise = np.random.normal(0, std, args.patch_dims)
                total_spectrograms += noise

        p = np.ascontiguousarray(total_spectrograms)
        p = torch.from_numpy(p).float()

    else:

        # put p between 0 and 1
        # p = p - p.min()
        # p = p / p.max()

        # augmentations
        if not eval:

            # with 50% chance to do noise addition
            if random.random() > -1:
                noise = np.random.normal(0, std, (args.patch_classes, args.seq_length))
                p += noise

            # 50% for each of the separate augmenations in augmenter
            # for i in range(args.patch_classes):
            #     p[i, :] = augmenter.augment(p[i, :])

        p = np.ascontiguousarray(p)
        p = torch.from_numpy(p).float().unsqueeze(0)

    # consider normalizing here!
    # p = Normalize(data_mean, data_std)(p)

    # output shape is (1, channels, sequence) typically (1, 8, 500)
    # or (8, 12, 12) for spectrogram
    return p


