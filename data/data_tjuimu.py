from os import path as osp
import numpy as np
from .base import IMUSequence
from mtools import csvread

def moving_average_filter(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def rerange_deg(raw_deg):
    # raw_deg = gt_df[bearing_key].values
    raw_deg = raw_deg%360 # 所有角度限制到[0,360]
    raw_deg[raw_deg>180] = raw_deg[raw_deg>180]-360
    return raw_deg

class TJUSequence(IMUSequence):
    feature_dim = 6
    label_dim = 2
    freq = 100

    def __init__(self, data_path, pose, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.labels, self.gt_pos = None, None, None, None
        self.info = {'path':osp.split(data_path), 'pose':pose, 'freq':self.freq}
        if data_path is not None:
            self.load(data_path)

    @staticmethod
    def get_data_list(root_dir, data_list, pose, mode):
        _data_list = list(filter(lambda x: True if pose== x.split('#')[2] else False, data_list))
        data_list_all = [osp.join(item, 'IMU_with_label_raw.txt') for item in _data_list]
        return data_list_all

    def load(self, path):
        imu_all = csvread(path, ' ')
        gyro = np.deg2rad(imu_all[:, 3:6])
        acce = imu_all[:, :3]/9.8
        acce_magn = np.linalg.norm(acce, axis=-1, keepdims=True)
        self.features = np.concatenate([gyro, acce, acce_magn], axis=1)
        self.labels = imu_all[:, 6:]

        enu_pos = np.column_stack((moving_average_filter(self.labels[:, 0], 10), moving_average_filter(self.labels[:, 1], 10)))
        yaws = rerange_deg(np.rad2deg(np.arctan2(enu_pos[1:, 1]-enu_pos[:-1, 1], enu_pos[1:, 0]-enu_pos[:-1, 0]))-90)
        yaws = np.concatenate((yaws, (yaws[0],)))
        self.oris = np.column_stack((yaws, yaws))
        print('shapes: ', self.features.shape, self.labels.shape)

    def get_feature(self):
        return self.features

    def get_label(self):
        return self.labels

    def get_ori(self):
        return self.oris

    def get_meta(self):
        return '{}: pose {}, freq {}Hz'.format(self.info['path'], self.info['pose'], self.info['freq'])
