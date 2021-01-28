# goes thru entire pipeline

from utils.model import Conv_Model
import torch
from myargs import args
import time
import numpy as np
from scipy import signal


class DeployModel:

    def __init__(self, model_path, channels=8, seq_length=500):

        self.ch = channels
        self.seq_len = seq_length

        # load model
        model = Conv_Model()
        # pretrained_dict = torch.load(model_path)['state_dict']
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        self.model = model.cuda().eval()

    def process_data(self, data, samp_freq=250, nperseg=32):
        # assume we receive ch*seq_len array, choose nperseg to make a square array
        spectro = []
        for cha in range(self.ch):
            # use n per seg of 32 to have a shape of 17x17
            f, t, spectrogram = signal.spectrogram(data[cha], fs=samp_freq, nperseg=nperseg)
            spectro.append(spectrogram)

        # finally convert to array
        spectro = np.asarray(spectro)

        # put spec between 0 and 1
        spectro = spectro - spectro.min()
        spectro = spectro / spectro.max()

        spectro = np.ascontiguousarray(spectro)
        spectro = torch.from_numpy(spectro).float().unsqueeze(0)
        return spectro

    def data_gen_test(self):
        # while True:
        # temp data out
        data = np.random.normal(loc=0, scale=1, size=(self.ch, self.seq_len))
        # df_ecg = cytonBoard.poll(seq_length)  # Polling for samples
        # data = df_ecg.iloc[:, 0].values  ## extracting values
        return data
        # time.sleep(1)  # Updating the window in every one second

    def get_data_and_model(self):

        # get generator
        x = self.data_gen_test()

        # no grad for eval
        with torch.no_grad():

            # send data and get model output
            proc_data = self.process_data(x).cuda()
            out = self.model(proc_data)
            print(out)
            out = torch.argmax(out, dim=1).item()

            data = [d.tolist() for d in x]

        return data, out


if __name__ == '__main__':
    dm = DeployModel('./data/model/spectro_conv_134.pt')
    # dm.deploy()
