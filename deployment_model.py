from scipy import signal
import argparse
import numpt as np
from numpy import genfromtxt
from utils.model import Simp_Model


parser = argparse.ArgumentParser()
parser.add_argument('--file_name', default='./file_name_here',
                    help='datafile name')


def prediction(): 
    
    num_channels=8
    
    args = parser.parse_args()

    data_array = genfromtxt(args.file_name, delimiter=',')
    data_array = data_array[:, 1:9]
    
    total_spectrograms = []
    for cha in range(num_channels):
    # use n per seg of 22 to have a shape of 12x12
        f, t, spectrogram = signal.spectrogram(data_array[:, cha], fs=250, nperseg=22)
        total_spectrograms.append(spectrogram)
    
    total_spectrograms = np.asarray(total_spectrograms)
     
    model = Simp_Model()
    model.load_state_dict(torch.load( NEED PATH HERE ))
    model.eval()
    
    image = total_spectrograms  
    image = image.flatten()
    prediction = model(image)
    
if __name__ == '__main__':
    prediction()
    