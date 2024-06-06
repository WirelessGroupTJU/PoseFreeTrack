from os import path as osp
import pandas
import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion

from .base import IMUSequence

class RIDISequence(IMUSequence):
    """
    Dataset :- RIDI (can be downloaded from https://wustl.app.box.com/s/6lzfkaw00w76f8dmu0axax7441xcrzd9)
    Features :- raw angular rate and acceleration (includes gravity).
    """

    feature_dim = 6
    label_dim = 2
    freq = 200

    def __init__(self, data_path, pose, **kwargs):

        super().__init__(**kwargs)
        self.ts, self.features, self.labels, self.gt_pos = None, None, None, None
        self.info = {'path':osp.split(data_path), 'pose':pose, 'freq':self.freq}
        if data_path is not None:
            self.load(data_path)

    @staticmethod
    def get_data_list(root_dir, data_list, pose, mode):
        _data_list = list(filter(lambda x: True if x.split('_')[1].startswith(pose) else False, data_list))
        data_list_all = [osp.join(item, 'processed', 'data.csv') for item in _data_list]
        return data_list_all

    def load(self, path):
        if path[-1] == '/':
            path = path[:-1]

        imu_all = pandas.read_csv(path)
        ts = imu_all[['time']].values / 1e09
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
        grav = imu_all[['grav_x', 'grav_y', 'grav_z']].values
        acce = (acce - grav) / 9.8
        acce_magn = np.linalg.norm(acce, axis=-1, keepdims=True)
        tango_pos = imu_all[['pos_x', 'pos_y']].values

        game_rv = quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)
        self.orientations = quaternion.as_float_array(game_rv)
        self.oris = R.from_quat(self.orientations).as_euler('yxz', degrees=True)

        self.ts = ts
        self.features = np.concatenate([gyro, acce, acce_magn], axis=1)
        self.labels = tango_pos
        print('shapes: ', self.ts.shape, self.features.shape, self.labels.shape)

    def get_feature(self):
        return self.features[::2]

    def get_label(self):
        return self.labels[::2]

    def get_ori(self):
        return self.oris[::2]

    def get_meta(self):
        return '{}: pose {}, freq {}Hz'.format(self.info['path'], self.info['pose'], self.info['freq']//2)
