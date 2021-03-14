from scipy import io
from glob import glob
import numpy as np
import os
from scipy.signal import butter, lfilter, spectrogram


def butter_bandpass(lowcut, highcut, fs=512, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass', output='ba')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, order=5):
    b, a = butter_bandpass(lowcut, highcut, order=order)
    y = lfilter(b, a, data)
    return y


def process_online(path, savepath, valpath):

    if not os.path.exists(savepath):
        os.mkdir(savepath)
    if not os.path.exists(valpath):
        os.mkdir(valpath)

    data_paths = glob(path + "*.mat")

    num_data = 0
    for datapath in data_paths:
        patient_data = io.loadmat(datapath)

        # sampling rate, should be 512 Hz
        srate = patient_data['eeg'][0][0][2][0][0]

        # left, right imagery data, shape should be (channels, time) = (68, big number)
        left_im = patient_data['eeg'][0][0][7]
        right_im = patient_data['eeg'][0][0][8]

        # imagery_events in (time)
        im_events = patient_data['eeg'][0][0][11][0]
        im_events = np.argwhere(im_events)[:, 0].tolist()

        # bad_trial_indices, just check if there are any
        bad_trial_idx_voltage = patient_data['eeg'][0][0][14][0][0][0][0]
        bad_trial_idx_mi = patient_data['eeg'][0][0][14][0][0][1][0]
        bad_trial_idx_voltage = bad_trial_idx_voltage[0].flatten().tolist() + bad_trial_idx_voltage[1].flatten().tolist()
        bad_trial_idx_mi = bad_trial_idx_mi[0].flatten().tolist() + bad_trial_idx_mi[1].flatten().tolist()
        allbad = bad_trial_idx_voltage + bad_trial_idx_mi

        # remove them from event list for left and right
        im_events = [event for ind, event in enumerate(im_events) if ind+1 not in set(allbad)]

        subject_num = datapath.split('\\')[-1].split('.')[0]

        val_split = int(0.8 * len(im_events))
        print(f'subject num: {subject_num}, events: {len(im_events)*2}, bad trials: {allbad}')
        print(im_events)

        # cut out for each im event
        for event_num, event_time in enumerate(im_events):

            # get 2 seconds of each piece of data 0.5s after stimulus, remove EMG channels
            data_left = left_im[0: 64, event_time + srate // 2: event_time + srate * 5 // 2]
            data_right = right_im[0: 64, event_time + srate // 2: event_time + srate * 5 // 2]

            # could cut -2.5 to -0.5s before stimulus to get a null example too for 3 class problem

            # filter here  =====  CAN ADJUST TO TRY DIFFERENT FILTERS AND STUFF
            # common average reference
            for t in range(srate * 2):
                data_left[:, t] = data_left[:, t] - data_left[:, t].mean()
                data_right[:, t] = data_right[:, t] - data_right[:, t].mean()

            # bandpass filter
            for i in range(len(data_left)):
                data_left[i] = butter_bandpass_filter(data_left[i], 8, 30, order=4)
                data_right[i] = butter_bandpass_filter(data_right[i], 8, 30, order=4)

            if event_num < val_split:
                # save train dicts for each datapoint
                np.save(f'{savepath}/{num_data}.npy', {'subject': subject_num, 'data': data_left, 'label': 'left'})
                np.save(f'{savepath}/{num_data+1}.npy', {'subject': subject_num, 'data': data_right, 'label': 'right'})
            else:
                # save train dicts for each datapoint
                np.save(f'{valpath}/{num_data}.npy', {'subject': subject_num, 'data': data_left, 'label': 'left'})
                np.save(f'{valpath}/{num_data + 1}.npy', {'subject': subject_num, 'data': data_right, 'label': 'right'})

            num_data += 2


if __name__ == '__main__':
    process_online('../data/eeg_online/', '../data/preproc_online_train/', '../data/preproc_online_val/')