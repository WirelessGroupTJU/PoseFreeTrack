from os import path as osp
import os
import numpy as np

from .base import IMUSequence
from mtools import read_file, join_path, csvread

list_dict = {
    'train': 'Train.txt',
    'test': 'Test.txt'
}

class OxIODSequence(IMUSequence):
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
            return OxIODSequence.get_data_list(root_dir, data_list, pose, 'train') + OxIODSequence.get_data_list(root_dir, data_list, pose, 'test')
        else:
            data_list = read_file(osp.join(root_dir, pose, list_dict[mode]))
            data_list = [data.split(osp.sep) for data in data_list]
            [data.insert(1, 'syn') for data in data_list]
            [data.insert(0, pose) for data in data_list]
            data_list = [osp.join(*data) for data in data_list]
            data_list_all = []
            for data in data_list:
                if osp.isdir(osp.join(root_dir, data)):
                    files = list(filter(lambda x:x.startswith('imu'), os.listdir(osp.join(root_dir, data))))
                    files = [osp.join(data, file) for file in files]
                    data_list_all.extend(files)
                elif osp.isfile(osp.join(root_dir, data)):
                    data_list_all.append(data)
                else:
                    print('%s is ignored'%(data))
        return data_list_all

    def load(self, path):
        if path[-1] == '/':
            path = path[:-1]
        imu_all = csvread(path)
        vi_all = csvread(path.replace('imu','vi'))
        gyro = imu_all[:, 4:7]
        acce = imu_all[:, 10:13]

        self.ts = vi_all[:, 1, np.newaxis]
        self.features = np.concatenate([gyro, acce], axis=1)
        self.labels = vi_all[:, 2:4]
        self.oris = vi_all[:, 5:9]
        
    def get_feature(self):
        return self.features

    def get_label(self):
        return self.labels
    
    def get_ori(self):
        return self.oris

    def get_meta(self):
        return '{}: pose {}, freq {}Hz'.format(self.info['path'], self.info['pose'], self.info['freq'])
