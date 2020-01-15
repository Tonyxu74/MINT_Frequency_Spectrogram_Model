import glob
from torch.utils import data
import numpy as np
import torch
from myargs import args


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, impath, eval):
        'Initialization'

        self.eval = eval

        # add to std if you would like some noise to be added to the spectrogram image
        self.std = 0

        # add images in different folders to the datalist
        datalist = []
        folders = glob.glob('{}/*/'.format(impath))

        for imfolder in folders:
            # open the gt.npy file in the folder and build a datalist
            label_dict = np.load('{}gt.npy'.format(imfolder), allow_pickle=True).flatten()[0]
            image_paths = label_dict.keys()
            datalist.append([
                {
                    'image': image_path,
                    'label': label_dict[image_path]
                } for image_path in image_paths
            ])
        self.datalist = [item for sublist in datalist for item in sublist]

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * 1 for i in self.datalist]))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'
        image = np.load(self.datalist[index]['image'].replace('../', './'))
        label = self.datalist[index]['label']

        image = normalizepatch(image, self.eval, self.std)

        return image, label


def GenerateIterator(args, impath, eval=False, shuffle=True):
    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(impath, eval=eval), **params)


def normalizepatch(p, eval, std):

    if not eval:
        noise = np.random.normal(0, std, args.patch_dims)
        p += noise

    p = np.ascontiguousarray(p)
    p = torch.from_numpy(p)

    # consider normalizing here!

    return p.float()


