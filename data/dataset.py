import ipdb as pdb
import math
from .data_ridi import RIDISequence
from .data_oxiod import OxIODSequence
from .data_oxiod_h5 import OxIODH5Sequence
from .data_tjuimu import TJUSequence
import random
import os.path as osp
import numpy as np
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, DataLoader
from .base import load_sequences
from .tools import get_polar_vectors, get_polar_labels, get_offset_vectors, rerange_angle, rerange_angle_value
from .math_util import rotate_data
from basictorch_v2.dataset import BDatasets, STDataLoader
from mtools import list_ind, list_mask, list_con, load_json, join_path

eps = 1e-5

def normalization(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    return (data - data_mean) / (data_std + eps)

def get_list(list_path, is_print=True):
    if list_path is not None:
        with open(list_path) as f:
            _list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        _list = None
    if _list is not None and is_print:
        print(_list)
    return _list

def get_dataset(root_dir, data_list, pose_list, args, mode='train', label_type='polar', target_pose_list=None, **kwargs):
    if mode == 'train':
        random_shift = args.step_size // 2 if 'random_shift' not in kwargs else kwargs['random_shift']
        shuffle = True if 'shuffle' not in kwargs else kwargs['shuffle']
        val_split = args.val_split
        data_usage=args.data_usage
    elif mode == 'test':
        random_shift = 0
        shuffle = False
        val_split = 0
        data_usage=1
    else:
        random_shift = 0
        shuffle = False
        val_split = 0
        data_usage=1
    
    if args.data_name == 'oxiod':
        seq_type = OxIODSequence
    elif args.data_name == 'oxiod_h5':
        seq_type = OxIODH5Sequence
    elif args.data_name == 'ridi':
        seq_type = RIDISequence
    elif args.data_name == 'tjuimu':
        seq_type = TJUSequence
    
    dataset = SequenceDataset(args, seq_type, label_type, root_dir, data_list, pose_list[args.source_pose], mode, random_shift=random_shift, shuffle=shuffle, data_usage=data_usage)
    if target_pose_list is not None:
        t_datasets = [SequenceDataset(args, seq_type, label_type, root_dir, data_list, pose, mode, random_shift=random_shift, shuffle=shuffle, data_usage=data_usage) for pose in target_pose_list]
        dataset = STSequenceDatasets(dataset, t_datasets)
    return dataset if val_split==0 else (dataset, dataset.get_split_dataset(val_split))

class SequenceDataset(Dataset):
    def __init__(self, args=None, seq_type=None, label_type=None, root_dir=None, data_list=None, pose=None, mode=None, random_shift=0, dataset=None, val_split=0, data_usage=1, **kwargs):
        super().__init__()
        if dataset is None:
            self.feature_dim = seq_type.feature_dim # 6
            self.label_dim = seq_type.label_dim     # 2
            self.pose = pose
            self.mode = mode
            self.random_shift = random_shift
            self.label_type = label_type
            self.window_size = args.window_size
            self.step_size = args.step_size
            self.data_freq = args.data_freq
            self.angle_stride = args.angle_stride
            self.angle_window = args.angle_window
            self.acc_magn = args.acc_magn
            self.yaw_diff = args.yaw_diff
            self.domain_std = args.domain_std
            self.yaw_diff_cur = args.yaw_diff_cur
            self.rho_threshold = getattr(args, 'rho_threshold', 0.0)
            self.index_map_list = []
            self.index_map = []
            self.seq_id = None
            self.domain_num = 1
            self.features, self.labels, self.oris, self.data_list = load_sequences(
                seq_type, root_dir, data_list, pose, mode, args.cache_path, args.data_freq, **kwargs)
            
            # if self.features[0].shape[1] == self.feature_dim:
            #     raise Warning('Maybe no acc_magn in features')
            
            # Optionally smooth the sequence
            if args.feature_sigma > 0:
                self.features = [gaussian_filter1d(feat, sigma=args.feature_sigma, axis=0) for feat in self.features]
            if args.label_sigma > 0:
                self.labels = [gaussian_filter1d(labe, sigma=args.label_sigma, axis=0) for labe in self.labels]
            
            if self.domain_std:
                self.domain_stats = load_json(args.domain_stats)
                dst = self.domain_stats[pose]
                mus = np.array([[0]*3 + [0]*3 + [dst[4]]])
                stds = np.array([[1]*3 + [1]*3 + [dst[5]]])
                self.features = [(feat-mus)/stds for feat in self.features]
            
            for i in range(len(self.data_list)):
                self.index_map_list.append([[i, j] for j in range(args.window_size*2+self.random_shift, self.labels[i].shape[0]-self.random_shift-self.angle_stride-1, args.step_size)])

            self.index_map = list_con(self.index_map_list)
            if kwargs.get('shuffle', True):
                random.shuffle(self.index_map)
            if self.rho_threshold>0:
                starts = [self.labels[seq_id][frame_id - self.window_size] for seq_id, frame_id in self.index_map]
                ends = [self.labels[seq_id][frame_id-1] for seq_id, frame_id in self.index_map]
                lengths_mask = [np.linalg.norm(end - start)<=self.rho_threshold for start, end in zip(starts, ends)]
                print('delete %d samples due to rho_threshold'%(len(lengths_mask)-np.sum(lengths_mask)))
                self.index_map = list_mask(self.index_map, lengths_mask)
            if data_usage<1:
                use_num = int(len(self.index_map)*data_usage)
                p = np.random.permutation(len(self.index_map))
                self.index_map = list_ind(self.index_map, p[:use_num])
        else:
            self.__dict__ = dataset.__dict__.copy()
            val_num = int(len(self.index_map)*val_split)
            p = np.random.permutation(len(self.index_map))
            self.index_map = list_ind(self.index_map, p[:val_num])
            dataset.index_map = list_ind(dataset.index_map, p[val_num:])
        
    def __getitem__(self, item):
        if self.seq_id is None:
            seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        else:
            seq_id, frame_id = self.seq_id, self.index_map_list[self.seq_id][item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)

        if self.acc_magn: # acc_magn at last column
            feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        else:
            feat = self.features[seq_id][frame_id - self.window_size:frame_id, :self.feature_dim]
            

        if self.yaw_diff:
            if self.yaw_diff_cur:
                yaw_init = self.oris[seq_id][frame_id - self.window_size, -1]
                yaw_diff = rerange_angle(self.oris[seq_id][frame_id - self.window_size:frame_id, [-1]] - yaw_init)
            else:
                yaw_diff = rerange_angle(self.oris[seq_id][frame_id - self.window_size:frame_id, [-1]] - self.oris[seq_id][frame_id - self.window_size*2:frame_id- self.window_size, [-1]])
            feat = np.concatenate((yaw_diff, feat), axis=-1)

        if self.label_type == 'polar':
            locs = self.labels[seq_id][frame_id - self.window_size*2:frame_id]
            init_psi, labe = get_polar_vectors(locs, self.window_size, [0, self.window_size])
            labe = labe[self.window_size:]
        else:
            locs = self.labels[seq_id][frame_id - self.window_size - self.angle_window//2 - self.angle_stride//2 : frame_id + self.angle_window//2 + self.angle_stride//2 + 1]
            init_psi, labe = get_polar_labels(locs, self.angle_window, self.angle_stride, phi=True)
        return feat.astype(np.float32), labe.astype(np.float32), locs.astype(np.float32), init_psi

    def __len__(self):
        return len(self.index_map) if self.seq_id is None else len(self.index_map_list[self.seq_id])
    
    def get_data_loader(self, batch_size=None, shuffle=False, seq_id=None):
        self.seq_id = seq_id
        return DataLoader(self, batch_size if batch_size else 1000, shuffle=shuffle)
    
    def get_labels(self):
        if self.seq_id is None:
            return self.labels
        else:
            return self.labels[self.seq_id]
        
    def get_split_dataset(self, val_split):
        return SequenceDataset(dataset=self, val_split = val_split)
    
    def print_shape(self):
        print('length: ', len(self))

class STSequenceDatasets(object):
    def __init__(self, s_dataset, t_datasets):
        self.s_dataset = s_dataset
        self.t_datasets = t_datasets
        self.domain_num = 1+len(t_datasets)
    
    def get_split_dataset(self, val_split):
        return STSequenceDatasets(SequenceDataset(dataset=self.s_dataset, val_split = val_split), 
            [SequenceDataset(dataset=dataset, val_split = val_split) for dataset in self.t_datasets])
    
    def __len__(self):
        return len(self.s_dataset)
    
    def get_data_loader(self, batch_size=None, shuffle=False):
        return STDataLoader([self.s_dataset.get_data_loader(batch_size, shuffle)], [dataset.get_data_loader(batch_size, shuffle) for dataset in self.t_datasets])
    
    def print_shape(self):
        print('source: ', len(self.s_dataset), 'target: ', ','.join([str(len(dataset)) for dataset in self.t_datasets]))
        # , ' feat_mean: ', self.s_dataset.feat_mean, ' feat_std: ', self.s_dataset.feat_std

class Datasets(BDatasets):
    def __init__(self, is_shuffle, is_validate, split_val, batch_size, pose_list, train_dataset=None, test_dataset=None, val_dataset=None, **kwargs):
        self.pose_list = pose_list
        self.domain_num = len(pose_list)
        super().__init__(is_shuffle, is_validate, split_val, batch_size, train_dataset=train_dataset, test_dataset=test_dataset, val_dataset=val_dataset, **kwargs)

    def print_shape(self, ds_names=['train_dataset', 'test_dataset', 'val_dataset']):
        for ds_name in ds_names:
            if ds_name in self.__dict__ and self.__dict__[ds_name] is not None:
                self.__dict__[ds_name].print_shape()
