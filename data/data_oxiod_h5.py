from os import path as osp
import os
import numpy as np
import pandas as pd
import ipdb as pdb

from .base import IMUSequence
from mtools import read_file, join_path, csvread

list_dict = {
    'train': 'Train.txt',
    'test': 'Test.txt'
}

class OxIODH5Sequence(IMUSequence):
    """
    Dataset     :
    Features    :
    """
    feature_dim = 6
    label_dim = 2
    freq = 100

    def __init__(self, data_path, pose, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.labels = None, None, None
        self.info = {'path':osp.split(data_path), 'pose':pose, 'freq':self.freq}
        if data_path is not None:
            self.load(data_path)
    
    @staticmethod
    def get_data_list(root_dir, data_list, pose, mode):
        if mode == 'all':
            return OxIODH5Sequence.get_data_list(root_dir, data_list, pose, 'train') + OxIODH5Sequence.get_data_list(root_dir, data_list, pose, 'test')
        else:
            data_list = read_file(osp.join(root_dir, pose, list_dict[mode]))
            data_list = list(filter(lambda x: not x.startswith('#'), data_list))
            data_list_all = [osp.join(pose, data) for data in data_list]
        return data_list_all

    def load(self, file_path):
        store = pd.HDFStore(file_path, mode='r')
        all_df = store.get('all_imu')
        store.close()

        gyro = all_df[['GysX', 'GysY', 'GysZ']].values.astype("float32")
        acce = all_df[['AccX', 'AccY', 'AccZ']].values.astype("float32")
        acce_magn = np.linalg.norm(acce, axis=-1, keepdims=True)

        self.ts = all_df['time'].values.astype("float32")
        self.features = np.concatenate([gyro, acce, acce_magn], axis=1)
        self.labels = all_df[['PosE', 'PosN']].values.astype("float32") # , 'PosU'
        self.oris = all_df[['rollDeg', 'pitchDeg', 'yawDeg']].values.astype("float32")
        
    def get_feature(self):
        return self.features

    def get_label(self):
        return self.labels
    
    def get_ori(self):
        return self.oris

    def get_meta(self):
        return '{}: pose {}, freq {}Hz'.format(self.info['path'], self.info['pose'], self.info['freq'])
