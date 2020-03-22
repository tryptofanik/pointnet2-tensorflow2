from glob import glob

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

IDX_LOOKUP = {'a.1': 0, 'a.100': 1, 'a.102': 2, 'a.104': 3, 'a.118': 4, 'a.121': 5, 'a.123': 6, 'a.128': 7, 'a.132': 8, 'a.137': 9, 'a.138': 10, 'a.140': 11, 'a.2': 12, 'a.211': 13, 'a.22': 14, 'a.24': 15, 'a.25': 16, 'a.26': 17, 'a.27': 18, 'a.28': 19, 'a.29': 20, 'a.3': 21, 'a.30': 22, 'a.35': 23, 'a.39': 24, 'a.4': 25, 'a.40': 26, 'a.43': 27, 'a.45': 28, 'a.47': 29, 'a.5': 30, 'a.6': 31, 'a.60': 32, 'a.7': 33, 'a.74': 34, 'a.77': 35, 'a.8': 36, 'a.80': 37, 'a.96': 38, 'b.1': 39, 'b.11': 40, 'b.121': 41, 'b.122': 42, 'b.18': 43, 'b.2': 44, 'b.22': 45, 'b.26': 46, 'b.29': 47, 'b.3': 48, 'b.30': 49, 'b.33': 50, 'b.34': 51, 'b.35': 52, 'b.36': 53, 'b.38': 54, 'b.40': 55, 'b.42': 56, 'b.43': 57, 'b.44': 58, 'b.45': 59, 'b.47': 60, 'b.49': 61, 'b.50': 62, 'b.52': 63, 'b.55': 64, 'b.6': 65, 'b.60': 66, 'b.61': 67, 'b.67': 68, 'b.68': 69, 'b.69': 70, 'b.7': 71, 'b.71': 72, 'b.72': 73, 'b.80': 74, 'b.81': 75, 'b.82': 76, 'b.84': 77, 'b.85': 78, 'b.92': 79, 'c.1': 80, 'c.10': 81, 'c.108': 82, 'c.116': 83, 'c.120': 84, 'c.124': 85, 'c.14': 86, 'c.2': 87, 'c.23': 88, 'c.25': 89, 'c.26': 90, 'c.3': 91, 'c.30': 92, 'c.31': 93, 'c.36': 94, 'c.37': 95, 'c.43': 96, 'c.45': 97, 'c.46': 98, 'c.47': 99, 'c.51': 100, 'c.52': 101, 'c.55': 102, 'c.56': 103, 'c.58': 104, 'c.6': 105, 'c.61': 106, 'c.62': 107, 'c.66': 108, 'c.67': 109, 'c.68': 110, 'c.69': 111, 'c.71': 112, 'c.72': 113, 'c.77': 114, 'c.78': 115, 'c.79': 116, 'c.8': 117, 'c.80': 118, 'c.82': 119, 'c.87': 120, 'c.92': 121, 'c.93': 122, 'c.94': 123, 'c.95': 124, 'c.97': 125, 'd.104': 126, 'd.108': 127, 'd.109': 128, 'd.110': 129, 'd.113': 130, 'd.122': 131, 'd.126': 132, 'd.129': 133, 'd.131': 134, 'd.136': 135, 'd.14': 136, 'd.142': 137, 'd.144': 138, 'd.145': 139, 'd.15': 140, 'd.153': 141, 'd.157': 142, 'd.159': 143, 'd.16': 144, 'd.162': 145, 'd.166': 146, 'd.169': 147, 'd.17': 148, 'd.185': 149, 'd.19': 150, 'd.198': 151, 'd.2': 152, 'd.20': 153, 'd.21': 154, 'd.211': 155, 'd.218': 156, 'd.26': 157, 'd.3': 158, 'd.32': 159, 'd.37': 160, 'd.38': 161, 'd.41': 162, 'd.50': 163, 'd.51': 164, 'd.52': 165, 'd.54': 166, 'd.68': 167, 'd.74': 168, 'd.79': 169, 'd.80': 170, 'd.81': 171, 'd.87': 172, 'd.9': 173, 'd.90': 174, 'd.92': 175, 'd.93': 176, 'd.95': 177, 'd.96': 178, 'e.1': 179, 'e.3': 180, 'e.6': 181, 'e.8': 182, 'f.1': 183, 'f.23': 184, 'f.4': 185, 'g.17': 186, 'g.18': 187, 'g.24': 188, 'g.3': 189, 'g.37': 190, 'g.39': 191, 'g.41': 192, 'g.44': 193, 'g.50': 194, 'g.68': 195, 'g.7': 196, 'g.9': 197}


def parse_filename(filename_batch, n_points=1024, cords_channels=3, features_channels=0, is_test=False):

    pt_clouds = []
    labels = []
    for filename in filename_batch:
        # Read in point cloud
        filename_str = filename.numpy().decode()
        pt_cloud = np.load(filename_str)
        ind = np.arange(pt_cloud.shape[0])
        if len(ind) > n_points:
            ind = np.random.choice(ind, n_points, replace=False)
        else:
            ind = np.random.choice(ind, n_points, replace=True)

        pt_cloud = pt_cloud[ind, :]

        # Add rotation and jitter to cords cloud
        if not is_test:
            cords_cloud = pt_cloud[:, :cords_channels]
            features_cloud = pt_cloud[:, cords_channels:]
            theta = np.random.random() * 2 * np.pi
            A = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
            offsets = np.random.normal(0, 0.02, size=cords_cloud.shape)
            cords_cloud = np.matmul(cords_cloud, A) + offsets
            pt_cloud = np.concatenate([cords_cloud, features_cloud], axis=1)

        # Create classification label
        obj_type = filename_str.split('/')[-3]   # e.g., airplane, bathtub
        label = np.zeros(len(IDX_LOOKUP), dtype=np.float32)
        label[IDX_LOOKUP[obj_type]] = 1.0

        pt_clouds.append(pt_cloud)
        labels.append(label)

    return np.stack(pt_clouds), np.stack(labels)


def tf_parse_filename(filename, parse_filename=parse_filename):
    """Take batch of filenames and create point cloud and label"""

    x, y = tf.py_function(parse_filename, [filename], [tf.float32, tf.float32])
    return x, y


def train_val_split(dataset_path, train_size=0.92):
    train, val = [], []
    for obj_type in glob(f'{dataset_path}/*/'):
        cur_files = glob(obj_type + 'train/*.npy')
        cur_train, cur_val = \
            train_test_split(cur_files, train_size=train_size, random_state=0, shuffle=True)
        train.extend(cur_train)
        val.extend(cur_val)

    return train, val
