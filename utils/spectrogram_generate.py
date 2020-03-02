from scipy import signal
from glob import glob
import datetime
import numpy as np
import os
#from myargs import args


def gen_spectrogram():
    # find all data folders
    raw_path_list = glob("../data/raw/")

    for raw_path in raw_path_list:

        # find raw data and logs separately
        data_paths = glob(raw_path + "raw*.txt")
        label_paths = glob(raw_path + "log*.txt")

        # make training folders IF they do not already exist
        if not os.path.exists(raw_path.replace('raw', 'train')):
            os.mkdir(raw_path.replace('raw', 'train'))

        # make the file linking each data item to the ground truth
        gt_file = {}

        # keep track of how many data points we generate PER experimentee
        num_data_points = 0

        for data_path, label_path in zip(data_paths, label_paths):
            # open each txt file
            data_file = open(data_path, "r")
            label_file = open(label_path, "r")
            print('Reading ' + str(data_path))

            # === extracting label info ===
            lines_read = 0
            label_list = []
            for line in label_file:
                # so we ignore the first 2 lines because they aren't the data yet
                if lines_read < 4:
                    lines_read += 1
                    continue
                label_val, t_in_vid, t_of_clip = line.split(' ')
                label_list.append({
                    'label': int(label_val) + 1,
                    'time': int(t_in_vid),
                    'length': int(t_of_clip.replace('\n', ''))
                })

                lines_read += 1
            # get some null values as well (guess that nothing is happening, occurs in gaps between labels)
            prev_time = 0
            for i in range(len(label_list)):
                label_list.append({
                    'label': 0,
                    'time': prev_time,
                    'length': label_list[i]['time'] - prev_time
                })
                prev_time = label_list[i]['time'] + label_list[i]['length']

            # === extracting raw data info ===
            lines_read = 0
            data_list = []
            for line in data_file:
                # skip first 6 lines (not data)
                if lines_read < 7:
                    lines_read += 1
                    continue

                # use some string manipulation to find time
                time = line.split(', ')[-2]
                #print(time)
                hr, min, sec = time.split(':')
                #print(min)
                hr, min = int(hr), int(min)
                sec, microsec = sec.split('.')
                sec, microsec = int(sec), int(microsec) * 1000

                # year month and day are arbitrary, use datetime library to add times quickly
                time = datetime.datetime(
                    year=2019,
                    month=12,
                    day=31,
                    hour=hr,
                    minute=min,
                    second=sec,
                    microsecond=microsec
                )

                # get the actual data point at this time
                data_pt = line.split(', ')[1:9]
                data_pt = [float(val) for val in data_pt]
                data_list.append({'val': data_pt, 'time': time})

                lines_read += 1

            # start time is first value in the datalist
            start_time = data_list[0]['time']

            # save a spectrogram for each label we have
            for label in label_list:
                # this gets value of the label
                gt_val = label['label']

                # this gets the time of when this value occurs, use the time it occurs (in seconds), add half of the
                # length of the clip to ensure that we gather data near the center of the clip
                time_of_clip = start_time + datetime.timedelta(seconds=(label['time'] + label['length']//2))

                data_num = 0
                data_clip = []
                for data in data_list:
                    # use this to determine how long of a snippet we want to train on (currently 1 second /
                    # 250 datapoints)
                    if data_num > 249:
                        break

                    # when we find the time that is just larger than the time we are looking for, begin data collection
                    # as above, data collection stops in exactly 250 datapoints
                    if data['time'] > time_of_clip:
                        data_clip.append(data['val'])
                        data_num += 1

                data_arr = np.asarray(data_clip)
                num_channels = 8

                # generate and append spectrograms of each channel
                total_spectrograms = []
                for cha in range(num_channels):
                    # use n per seg of 22 to have a shape of 12x12
                    f, t, spectrogram = signal.spectrogram(data_arr[:, cha], fs=250, nperseg=22)
                    total_spectrograms.append(spectrogram)

                # finally convert to array and save
                total_spectrograms = np.asarray(total_spectrograms)
                save_path = raw_path.replace('raw', 'train') + '{}.npy'.format(num_data_points)
                gt_file['{}.npy'.format(num_data_points)] = gt_val
                np.save(save_path, total_spectrograms)

                # we have saved a new datapoint, this increment this by one
                num_data_points += 1

        # save the gt info file
        np.save(raw_path.replace('raw', 'train') + 'gt.npy', gt_file)


if __name__ == "__main__":
    gen_spectrogram()
