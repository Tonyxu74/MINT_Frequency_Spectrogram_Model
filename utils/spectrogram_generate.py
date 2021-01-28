from scipy import signal
from glob import glob
import datetime
import numpy as np
import os
from myargs import args
from tqdm import tqdm
from collections import Counter


def gen_spectrogram():
    # find all data folders
    raw_path_list = ["../data/raw/"]

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

            # === extracting label info ===
            lines_read = 0
            label_list = []
            for line in label_file:
                # so we ignore the first 2 lines because they aren't the data yet
                if lines_read < 3:
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

            print(len(label_list), data_path, label_path)

            # === extracting raw data info ===
            lines_read = 0
            data_list = []
            for line in data_file:
                # skip first 7 lines (not data)
                if lines_read < 7:
                    lines_read += 1
                    continue

                # use some string manipulation to find time
                time = line.split(', ')[-2]
                hr, min, sec = time.split(':')
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
                time_of_clip = start_time + datetime.timedelta(
                    seconds=(label['time'] + label['length']/2 - args.seq_length/250/2)
                )

                data_num = 0
                data_clip = []
                for data in data_list:
                    # use this to determine how long of a snippet we want to train on (currently 1 second /
                    # seq_length datapoints)
                    if data_num > args.seq_length - 1:
                        break

                    # when we find the time that is just larger than the time we are looking for, begin data collection
                    # as above, data collection stops in exactly 250 datapoints
                    if data['time'] > time_of_clip:
                        data_clip.append(data['val'])
                        data_num += 1

                data_arr = np.asarray(data_clip)
                data_arr = np.transpose(data_arr, (1, 0))

                # # this is for spectrogram data
                # num_channels = 8
                # # generate and append spectrograms of each channel
                # total_spectrograms = []
                # for cha in range(num_channels):
                #     # use n per seg of 22 to have a shape of 12x12
                #     f, t, spectrogram = signal.spectrogram(data_arr[:, cha], fs=250, nperseg=22)
                #     total_spectrograms.append(spectrogram)
                #
                # # finally convert to array and save
                # total_spectrograms = np.asarray(total_spectrograms)
                #
                # print(total_spectrograms.shape)

                print(data_arr.shape)

                save_path = raw_path.replace('raw', 'train') + '{}.npy'.format(num_data_points)
                gt_file['{}.npy'.format(num_data_points)] = gt_val
                np.save(save_path, data_arr)

                # we have saved a new datapoint, this increment this by one
                num_data_points += 1

        # save the gt info file
        np.save('../gt.npy', gt_file)


def get_mean_std(path):

    list_datapaths = glob('{}/*/*.npy'.format(path))
    print(list_datapaths)
    values = []
    for path in list_datapaths:
        if 'gt' in path:
            continue
        # open an image
        img = np.load(path, allow_pickle=True).flatten()[0]
        np.asarray(img)

        values.append(img)

    values = np.asarray(values)

    # return a mean and standard deviation
    return np.mean(values), np.std(values)


def resize_pretraining(datalist, labels):

    # get resize length
    resize_length = int(250 / 200 * len(datalist))

    # resizes from 200Hz to 250Hz
    res_data, res_labels = [], []

    # j counts up to the total number in new sequence (args.seq_length)
    j = 0

    # iterate through the frames
    for i, (data, label) in enumerate(zip(datalist, labels)):

        # since the length is shorter, add the frame
        res_data.append(data)
        res_labels.append(label)
        j += 1

        # now, continue adding duplicates of the item based on the ratio between 250 and 200 Hz
        while int(float(j) / resize_length * len(datalist)) == i:
            res_data.append(data)
            res_labels.append(label)
            j += 1

    # check sizes
    assert len(res_data) == resize_length and len(res_labels) == resize_length
    return res_data, res_labels


def gen_pretrain_data():

    path = '../data/txt/'

    # make path to pretrain folder if not exist
    if not os.path.exists(path.replace('txt', 'pretrain')):
        os.mkdir(path.replace('txt', 'pretrain'))

    # grab paths to each data point
    label_paths = glob(path + '*log.txt')
    data_paths = [p.replace('_log', '') for p in label_paths]

    gt_file = {}
    num_data = 0

    # 0 is passive, 1 is left arm, 2 is left leg, 3 is right arm, 4 is right leg
    gt_map = {1: 1, 2: 3, 3: 0, 4: 2, 5: 0, 6: 4}

    for data_path, label_path in zip(data_paths, label_paths):

        print(data_path, label_path)
        data_file = open(data_path, "r")
        label_file = open(label_path, "r")

        data_list = []
        gt_list = []
        for data_pt, label in zip(data_file, label_file):
            data_pt = [float(pt) for pt in data_pt.split(' ')]
            label = int(label)

            data_list.append(data_pt)
            gt_list.append(label)

        # resample data size to 250Hz
        print(f'original length of data: {len(data_list)}')
        data_list, gt_list = resize_pretraining(data_list, gt_list)
        print(f'resized length of data: {len(data_list)}')

        # iterate through everything
        for start_index in tqdm(range(0, len(data_list) - args.seq_length, args.seq_stride)):

            # get gt slice
            gt_slice = gt_list[start_index: start_index + args.seq_length]

            # get the label and how much of the slice equals the label
            count_gts = Counter(gt_slice)

            # if most common class is null
            if count_gts.most_common(1)[0][0] == 0:

                # check how many of null class there are, if all null, gt is 100% equal to 0
                if count_gts.most_common(1)[0][1] == args.seq_length:
                    gt = 0
                    percent_gt = 1.0

                # otherwise, check second most common class, and get how much percent it isn't null
                else:
                    gt = count_gts.most_common(2)[1][0]
                    percent_gt = count_gts.most_common(2)[1][1] / args.seq_length

            # another class for most common
            else:
                gt = count_gts.most_common(1)[0][0]
                percent_gt = count_gts.most_common(1)[0][1] / args.seq_length

            # skip if null class (no reading) or percent less than 20% (not enough label in example)
            # this 99 label only happens in one of the pieces of data and we can just ignore it if it comes up
            if gt == 0 or percent_gt < 0.2 or gt not in gt_map.keys():
                continue

            # get data slice
            data_slice = np.asarray(data_list[start_index: start_index + args.seq_length], dtype=np.float)
            data_slice = np.transpose(data_slice, (1, 0))

            # map each ground truth to their value in our dataset (their zeroes are actually nothing happening)
            gt = gt_map[gt]

            # save the stuff and add label to gt list, get number of example for save path
            gt_file['{}.npy'.format(num_data)] = gt
            save_path = path.replace('txt', 'pretrain') + '{}.npy'.format(num_data)
            np.save(save_path, data_slice)

            num_data += 1

    np.save('../pretrain_gt.npy', gt_file)


def han_data():
    # find all data folders
    raw_path_list = ["../data/20201117_RealMove_ImaginaryMove_Han/"]

    for raw_path in raw_path_list:

        # find raw data and logs separately
        data_paths = glob(raw_path + "raw_nov*.txt")
        label_paths = glob(raw_path + "log_nov*.txt")

        # make data folders IF they do not already exist
        if not os.path.exists(raw_path.replace('20201117_RealMove_ImaginaryMove_Han', 'htrain')):
            os.mkdir(raw_path.replace('20201117_RealMove_ImaginaryMove_Han', 'htrain'))

        if not os.path.exists(raw_path.replace('20201117_RealMove_ImaginaryMove_Han', 'hval')):
            os.mkdir(raw_path.replace('20201117_RealMove_ImaginaryMove_Han', 'hval'))

        # make the file linking each data item to the ground truth
        gt_file = {}

        # keep track of how many data points we generate PER experimentee
        num_data_points = 0

        for data_path, label_path in zip(data_paths, label_paths):

            # open each txt file
            data_file = open(data_path, "r")
            label_file = open(label_path, "r")

            # grab the c1, or c2, or c-whatever and trial number
            vid_num = data_path.split('_')[-3]
            trial_num = int(data_path.split('_')[-1].split('.')[0])
            timestamp_path = glob(f'{raw_path}log_timestamp*{vid_num}_*.txt')[0]
            timestamp_file = open(timestamp_path, "r")
            video_start = timestamp_file.readlines()[trial_num - 1].split(' ')[-1]

            # string for video start time
            hr, min, sec = video_start.split(':')
            hr, min = int(hr), int(min)
            sec, microsec = sec.split('.')
            sec, microsec = int(sec), int(microsec)

            # datetime object for video start time, since it's in one day (not at 12am) we can assign any yr/mon/day
            video_start = datetime.datetime(
                year=2019,
                month=12,
                day=31,
                hour=hr,
                minute=min,
                second=sec,
                microsecond=microsec
            )

            # === extracting label info ===
            lines_read = 0
            label_list = []
            for line in label_file:
                # so we ignore the first 2 lines because they aren't the data yet
                if lines_read < 3:
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

            print(f'num labels {len(label_list)}, paths: {data_path} {label_path} {timestamp_path}')

            # === extracting raw data info ===
            lines_read = 0
            data_list = []
            for line in data_file:
                # skip first 7 lines (not data)
                if lines_read < 7:
                    lines_read += 1
                    continue

                # use some string manipulation to find time
                time = line.split(', ')[-2]
                hr, min, sec = time.split(':')
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

                # make first datapoint in list when video starts
                if time < video_start:
                    continue

                # get the actual data point at this time
                data_pt = line.split(', ')[1:9]
                data_pt = [float(val) for val in data_pt]
                data_list.append({'val': data_pt, 'time': time})

                lines_read += 1

            # start time is first value in the datalist
            start_time = data_list[0]['time']
            print(f'start time: {start_time}, video start: {video_start}')

            # save a spectrogram for each label we have
            for label in label_list:
                # this gets value of the label
                gt_val = label['label']

                # this gets the time of when this value occurs, use the time it occurs (in seconds), add half of the
                # length of the clip to ensure that we gather data near the center of the clip
                # time_of_clip = start_time + datetime.timedelta(
                #     seconds=(label['time'] + label['length']/2 - args.seq_length/250/2)
                # )
                time_of_clip = label['time'] + label['length']/2 - args.seq_length/250/2

                data_num = 0
                data_clip = []
                for data in data_list[int(time_of_clip*250):]:
                    # use this to determine how long of a snippet we want to train on (currently 1 second /
                    # seq_length datapoints)
                    if data_num > args.seq_length - 1:
                        break

                    # when we find the time that is just larger than the time we are looking for, begin data collection
                    # as above, data collection stops in exactly 250 datapoints
                    # if data['time'] > time_of_clip:
                    #     data_clip.append(data['val'])
                    #     data_num += 1
                    data_clip.append(data['val'])
                    data_num += 1

                data_arr = np.asarray(data_clip)
                data_arr = np.transpose(data_arr, (1, 0))

                # # this is for spectrogram data
                # num_channels = 8
                # # generate and append spectrograms of each channel
                # total_spectrograms = []
                # for cha in range(num_channels):
                #     # use n per seg of 22 to have a shape of 12x12
                #     f, t, spectrogram = signal.spectrogram(data_arr[:, cha], fs=250, nperseg=22)
                #     total_spectrograms.append(spectrogram)
                #
                # # finally convert to array and save
                # total_spectrograms = np.asarray(total_spectrograms)
                #
                # print(total_spectrograms.shape)

                print(data_arr.shape)

                if trial_num == 3:
                    save_path = raw_path.replace(
                        '20201117_RealMove_ImaginaryMove_Han', 'hval') + '{}.npy'.format(num_data_points)
                elif trial_num == 1 or trial_num == 2:
                    save_path = raw_path.replace(
                        '20201117_RealMove_ImaginaryMove_Han', 'htrain') + '{}.npy'.format(num_data_points)
                else:
                    raise ValueError('Unexpected trial number!')

                gt_file['{}.npy'.format(num_data_points)] = gt_val
                np.save(save_path, data_arr)

                # we have saved a new datapoint, this increment this by one
                num_data_points += 1

        # save the gt info file
        np.save('../h_gt.npy', gt_file)


if __name__ == "__main__":
    # print(get_mean_std('../data/time_train'))
    # gen_spectrogram()
    # gen_pretrain_data()
    han_data()
