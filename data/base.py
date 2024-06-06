import warnings, json, h5py
import numpy as np
from os import path as osp
from abc import ABC, abstractmethod
from mtools import check_dir
from .math_util import gyro_integration

"""
We use two levels of hierarchy for flexible data loading pipeline:
  - Sequence: Read the sequence from file and compute per-frame feature and label.
  - Dataset: subclasses of PyTorch's Dataset class. It has three roles:
      1. Create a Sequence instance internally to load data and compute feature/label.
      2. Apply post processing, e.g. smoothing or truncating, to the loaded sequence.
      3. Define how to extract samples from the sequence.


To define a new dataset for training/testing:
  1. Subclass CompiledSequence class. Load data and compute feature/label in "load()" function.
  2. Subclass the PyTorch Dataset. In the constructor, use the custom CompiledSequence class to load data. You can also
     apply additional processing to the raw sequence, e.g. smoothing or truncating. Define how to extract samples from 
     the sequences by overriding "__getitem()__" function.
  3. If the feature/label computation are expensive, consider using "load_cached_sequence" function.
  
Please refer to GlobalSpeedSequence and DenseSequenceDataset in data_global_speed.py for reference. 
"""

class IMUSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_label(self):
        pass

    def get_meta(self):
        return "No info available"

def load_sequences(seq_type, root_dir, data_list, pose, mode, cache_path, freq, **kwargs):
    if cache_path is not None and cache_path not in ['none', 'invalid', 'None']:
        check_dir(cache_path)
        if osp.exists(osp.join(cache_path, 'config.json')):
            info = json.load(open(osp.join(cache_path, 'config.json')))
            if info['feature_dim'] != seq_type.feature_dim or info['label_dim'] != seq_type.label_dim:
                warnings.warn('The cached dataset has different feature or label dimension. Ignore')
                cache_path = 'invalid'
        else:
            info = {'feature_dim': seq_type.feature_dim, 'label_dim': seq_type.label_dim}
            json.dump(info, open(osp.join(cache_path, 'config.json'), 'w'))
    
    
    interval = int(seq_type.freq / freq)
    data_list = seq_type.get_data_list(root_dir, data_list, pose, mode)
    print('pose', mode, data_list)

    features_all, labels_all, oris_all = [], [], []
    for i in range(len(data_list)):
        if cache_path is not None and osp.exists(osp.join(cache_path, data_list[i] + '.hdf5')):
            with h5py.File(osp.join(cache_path, data_list[i] + '.hdf5'), 'r') as f:
                feat = np.copy(f['feature'])
                labe = np.copy(f['label'])
                if 'ori' in f:
                    ori  = np.copy(f['ori'])
                else:
                    ori = None
        else:
            seq = seq_type(osp.join(root_dir, data_list[i]), pose, **kwargs)
            feat, labe, ori = seq.get_feature(), seq.get_label(), seq.get_ori()
            print(seq.get_meta())
            if cache_path is not None and osp.isdir(cache_path):
                check_dir(osp.split(osp.join(cache_path, data_list[i] + '.hdf5'))[0])
                with h5py.File(osp.join(cache_path, data_list[i] + '.hdf5'), 'x') as f:
                    f['feature'] = feat
                    f['label'] = labe
                    if ori is not None:
                        f['ori'] = ori
                f.close()
        feat = feat[::interval]
        labe = labe[::interval]
        if ori is not None:
            ori = ori[::interval]
        features_all.append(feat)
        labels_all.append(labe)
        oris_all.append(ori)
    return features_all, labels_all, oris_all, data_list

def select_orientation_source(data_path, max_ori_error=20.0, grv_only=True, use_ekf=True):
    """
    Select orientation from one of gyro integration, game rotation vector or EKF orientation.

    Args:
        data_path: path to the compiled data. It should contain "data.hdf5" and "info.json".
        max_ori_error: maximum allow alignment error.
        grv_only: When set to True, only game rotation vector will be used.
                  When set to False:
                     * If game rotation vector's alignment error is smaller than "max_ori_error", use it.
                     * Otherwise, the orientation will be whichever gives lowest alignment error.
                  To force using the best of all sources, set "grv_only" to False and "max_ori_error" to -1.
                  To force using game rotation vector, set "max_ori_error" to any number greater than 360.


    Returns:
        source_name: a string. One of 'gyro_integration', 'game_rv' and 'ekf'.
        ori: the selected orientation.
        ori_error: the end-alignment error of selected orientation.
    """
    ori_names = ['gyro_integration', 'game_rv']
    ori_sources = [None, None, None]

    with open(osp.join(data_path, 'info.json')) as f:
        info = json.load(f)
        ori_errors = np.array(
            [info['gyro_integration_error'], info['grv_ori_error'], info['ekf_ori_error']])
        init_gyro_bias = np.array(info['imu_init_gyro_bias'])

    with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
        ori_sources[1] = np.copy(f['synced/game_rv'])
        if grv_only or ori_errors[1] < max_ori_error:
            min_id = 1
        else:
            if use_ekf:
                ori_names.append('ekf')
                ori_sources[2] = np.copy(f['pose/ekf_ori'])
            min_id = np.argmin(ori_errors[:len(ori_names)])
            # Only do gyro integration when necessary.
            if min_id == 0:
                ts = f['synced/time']
                gyro = f['synced/gyro_uncalib'] - init_gyro_bias
                ori_sources[0] = gyro_integration(ts, gyro, ori_sources[1][0])

    return ori_names[min_id], ori_sources[min_id], ori_errors[min_id]
