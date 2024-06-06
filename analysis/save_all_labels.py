import h5py
import os.path as osp
import numpy as np

from data.dataset import get_dataset, get_list
from basictorch_v2.tools import t2n

class Args(object):
    data_name = 'oxiod'
    root_dir = 'data/OxIOD'
    pose_list = 'configs/lists/oxiod_pose_list.txt'
    cache_path = 'data/cache/OxIOD/train_dataset_cache'
    data_freq = 100
    window_size = 200
    step_size = 10
    val_split = 0
    data_usage = 1.0
    source_pose = 0
    label_type = 'polar'
    angle_stride = 30
    angle_window = 10
    acc_magn = False
    yaw_diff = False
    domain_std = False
    feature_sigma = 0.00001
    label_sigma = 0

def get_single_labels(dataset):
    y_list_seq = []
    for i in range(len(dataset.index_map_list)):
        y_seq = []
        for _,batch_data in enumerate(dataset.get_data_loader(100, seq_id=i)):
            _, _y, locs, init_psi = batch_data
            _y = t2n(_y[:, -1, :])
            y_seq.append(_y)
        y_seq = np.concatenate(y_seq, 0)
        y_list_seq.append(y_seq)
    return y_list_seq

def save_labels(h5_name, y_lists, y_names=None):
    if y_names is None:
        y_names = [str(i) for i in range(len(y_lists))]
    with h5py.File(osp.join('analysis/labels', h5_name), 'w') as f:
        for i,_y_list in enumerate(y_lists):
            f[y_names[i].replace('/', '_')] = _y_list
            print(_y_list.shape)
        f.close()

args = Args()
pose_list = get_list(args.pose_list)
target_pose_list = pose_list.copy()
target_pose_list.pop(args.source_pose)
train_dataset = get_dataset(args.root_dir, None, pose_list, args, mode='all', label_type=args.label_type, target_pose_list=target_pose_list, random_shift=0, shuffle=False)
y_lists = []
y_list_seq = get_single_labels(train_dataset.s_dataset)
print(train_dataset.s_dataset.data_list)
save_labels('labels_%s.h5'%pose_list[0], y_list_seq, train_dataset.s_dataset.data_list)
y_lists.append(np.concatenate(y_list_seq, 0))
for i,t_dataset in enumerate(train_dataset.t_datasets):
    y_list_seq = get_single_labels(t_dataset)
    save_labels('labels_%s.h5'%pose_list[i+1], y_list_seq, t_dataset.data_list)
    y_lists.append(np.concatenate(y_list_seq, 0))
save_labels('labels.h5', y_lists)