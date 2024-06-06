import ipdb as pdb
import torch
# import h5py
# import random
import numpy as np
# import json
# import math
# import quaternion
# import os
# from os import path as osp
# import sys
from mtools import csvwrite, save_h5
import mtools.monkey as mk
import matplotlib.pyplot as plt
from math import ceil
from scipy import signal
from basictorch_v2.tools import t2n, data_to_device # , torchDevice, n2t
from basictorch_v2.losses import loss_funcs, eps

def set_args(args):
    pose_lists  = {'oxiod':'configs/lists/oxiod_pose_list.txt', 'oxiod_h5':'configs/lists/oxiod_pose_list.txt', 'ridi':'configs/lists/ridi_pose_list.txt', 'tjuimu':'configs/lists/tjuimu_pose_list.txt'}
    data_lists  = {'oxiod':None, 'oxiod_h5':None, 'ridi':'configs/lists/ridi_train_list.txt', 'tjuimu':'configs/lists/tjuimu_train_list.txt'}
    test_data_lists  = {'oxiod':None, 'oxiod_h5':None, 'ridi':'configs/lists/ridi_test_list.txt', 'tjuimu':'configs/lists/tjuimu_test_list.txt'}
    root_dirs   = {'oxiod':'data/OxIOD', 'oxiod_h5':'data/OxIOD_h5', 'ridi':'data/ridi', 'tjuimu':'data/tjuimu'}
    cache_paths = {'oxiod':'data/cache/OxIOD/train_dataset_cache', 'oxiod_h5':None, 'ridi':'data/cache/ridi/train_dataset_cache', 'tjuimu':'data/cache/tjuimu/train_dataset_cache'}
    domain_stats = {'oxiod_h5': 'data/data_conf/oxiod_domain_stat.json', 'ridi':'data/data_conf/ridi_domain_stat.json', 'tjuimu':'data/data_conf/tjuimu_domain_stat.json'}
    rho_thresholds = {'oxiod_h5': 2.1, 'ridi': 2.8, 'tjuimu': 2.8}
    
    args.pose_list = pose_lists[args.data_name]
    args.data_list = data_lists[args.data_name]
    args.test_data_list = test_data_lists[args.data_name]
    args.root_dir = root_dirs[args.data_name]
    args.cache_path = cache_paths[args.data_name]
    args.domain_stats = domain_stats[args.data_name] if args.data_name in domain_stats else None
    args.rho_threshold = rho_thresholds[args.data_name]

def remove_none_from_losses(losses):
    keys = list(losses.keys())
    for k in keys:
        if losses[k] is None:
            losses.pop(k)
    return losses

def gram(a):
    return torch.matmul(a, torch.swapaxes(a, -1, -2)) / a.shape[2]

def polar_to_offset(polars, is_torch=False):
    if is_torch:
        return torch.stack((polars[...,0]*torch.cos(polars[...,1]), polars[...,0]*torch.sin(polars[...,1])), dim=-1)
    else:
        return np.stack((polars[...,0]*np.cos(polars[...,1]), polars[...,0]*np.sin(polars[...,1])), axis=-1)

def offset_to_polar(offsets, is_torch=False):
    if is_torch:
        lengths = torch.norm(offsets, dim=-1)
        angles = torch.atan2(offsets[...,1], offsets[...,0])
        return torch.stack((lengths, angles), dim=-1)
    else:
        lengths = np.linalg.norm(offsets, axis=-1)
        angles = np.arctan2(offsets[...,1], offsets[...,0])
        return np.stack((lengths, angles), axis=-1)
        
def clip_angle(a):
    mask = a<-np.pi
    a[mask] += 2*np.pi
    mask = a>np.pi
    a[mask] -= 2*np.pi
    return a

def rerange_angle(rad):
    # raw_deg = gt_df[bearing_key].values
    deg = np.rad2deg(rad)%360 # 所有角度限制到[0,360]
    deg[deg>180] = deg[deg>180]-360
    return np.deg2rad(deg)

def rerange_angle_value(rad):
    deg = np.rad2deg(rad)%360 # 所有角度限制到[0,360]
    if deg>180:
        deg = deg-360
    return np.deg2rad(deg)

def get_polar_vectors(locations, origin=None, zero_ind=None, is_torch=False, reverse=False, zero_thres=0.1):
    offsets = locations-locations[...,origin,:] if origin is not None else locations
    if reverse:
        offsets = -offsets
    if is_torch:
        lengths = torch.norm(offsets, dim=-1)
        angles = torch.atan2(offsets[...,1], offsets[...,0])
        if zero_ind is not None:
            zero_vector = locations[...,zero_ind[1],:] - locations[...,zero_ind[0],:]
            zero_angle = torch.atan2(zero_vector[1], zero_vector[0])
            angles = rerange_angle(angles - zero_angle)
        if lengths[-1] < zero_thres:
            angles = torch.zeros_like(lengths)
        return zero_angle, torch.stack((lengths, angles), dim=1)
    else:
        lengths = np.linalg.norm(offsets, axis=-1)
        angles = np.arctan2(offsets[...,1], offsets[...,0])
        if zero_ind is not None:
            zero_vector = locations[...,zero_ind[1],:] - locations[...,zero_ind[0],:]
            zero_angle = np.arctan2(zero_vector[1], zero_vector[0])
            angles = rerange_angle(angles - zero_angle)
        if lengths[-1] < zero_thres:
            angles = np.zeros_like(lengths)
        return zero_angle, np.stack((lengths, angles), axis=1)

def get_offset_vectors(locations, origin=None, zero_ind=None, is_torch=False, reverse=False):
    _, polars = get_polar_vectors(locations, origin, zero_ind, is_torch, reverse)
    return polar_to_offset(polars, is_torch)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_polar_labels(locations, angle_window, angle_stride, phi=False, is_torch=False, zero_thres=0.1):
    module = torch if is_torch else np
    init_offsets = locations[...,angle_window//2+angle_stride//2:-angle_window//2-angle_stride//2,:]-locations[...,angle_window//2+angle_stride//2,:]
    lengths = torch.norm(init_offsets, dim=-1) if is_torch else np.linalg.norm(init_offsets, axis=-1)
    stride_offsets = locations[...,angle_stride:,:]-locations[...,:-angle_stride,:]
    headings = module.atan2(stride_offsets[...,1], stride_offsets[...,0]) if is_torch else module.arctan2(stride_offsets[...,1], stride_offsets[...,0])
    headings_diff = headings[1:] - headings[:-1]
    cut_indexs = module.where(module.abs(headings_diff)>np.pi)[0]
    for ind in cut_indexs:
        if headings_diff[ind]>0:
            headings[ind+1:] -= 2*np.pi
        else:
            headings[ind+1:] += 2*np.pi
    headings = moving_average(headings, angle_window+1)
    if lengths[-1] < zero_thres:
        init_psi = 0
        psi_angles = torch.zeros_like(lengths) if is_torch else np.zeros_like(lengths)
    else:
        init_psi = headings[0]
        psi_angles = rerange_angle(headings - init_psi)
    if lengths[-1] < zero_thres:
        phi_angles = torch.zeros_like(lengths) if is_torch else np.zeros_like(lengths)
    else:
        phi_angles = module.atan2(init_offsets[...,1], init_offsets[...,0]) if is_torch else module.arctan2(init_offsets[...,1], init_offsets[...,0])
        phi_angles = rerange_angle(phi_angles - init_psi)
    if phi:
        return init_psi, module.column_stack((lengths, psi_angles, phi_angles))[..., 1:,:]
    else:
        return init_psi, module.column_stack((lengths, phi_angles))[..., 1:,:]

def get_cos_c(a, b, theta, is_torch=False):
    if is_torch:
        return torch.sqrt(a**2+b**2-2*a*b*torch.cos(theta))
    else:
        return np.sqrt(a**2+b**2-2*a*b*np.cos(theta))

def get_theta(a, b, is_torch=False):
    if is_torch:
        T = torch
    else:
        T = np
    theta = T.abs(a-b)
    mask = theta<T.pi
    theta[mask] = T.pi - theta[mask]
    theta[~mask] -= T.pi
    return theta

def cross(x, y):
    return x[...,1]*y[...,0]-x[...,0]*y[...,1]

def get_projection_ld(offset, offset_p, is_torch=False):
    if is_torch:
        k = torch.sum(-offset_p*offset, dim=-1)/-torch.square(torch.norm(offset, dim=-1)+eps)
        dropFoot = torch.unsqueeze(k, -1) * offset
        l = torch.norm(dropFoot, dim=-1)
        d = torch.abs(cross(offset_p, offset_p - offset))/(torch.norm(offset, dim=-1)+eps)
    else:
        k = np.sum(-offset_p*offset, axis=-1)/-np.square(np.linalg.norm(offset, axis=-1)+eps)
        dropFoot = np.expand_dims(k, -1) * offset
        l = np.linalg.norm(dropFoot, axis=-1)
        d = np.abs(cross(offset_p, offset_p - offset))/(np.linalg.norm(offset, axis=-1)+eps)
    return l, d

def rotate_locations(locations, angle, is_torch=False):
    if is_torch:
        rotate_matrix = torch.tensor([[torch.cos(angle),-torch.sin(angle)],[torch.sin(angle),torch.cos(angle)]], device=locations.device)
        return torch.mm(locations, rotate_matrix)
    else:
        rotate_matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        return np.dot(locations, rotate_matrix)

def pred_loss(y, y_p, label_mode='sparse', loss_func='mse', trans_func=None, is_torch=False, step_size=1):
    if label_mode == 'sparse':
        _y, _yp = y[:,-1], y_p[:,-1]
    elif label_mode == 'dense':
        _y, _yp = y[:, step_size-1::step_size], y_p[:, step_size-1::step_size]
    elif label_mode == 'mix':
        return pred_loss(y, y_p, 'dense', loss_func, trans_func, is_torch, step_size) + \
            pred_loss(y, y_p, 'sparse', loss_func, trans_func, is_torch, step_size)
    else:
        raise Exception('Unexpected label_mode: %s' % label_mode)
    
    if trans_func is not None:
        _y, _yp = trans_func(_y, is_torch=is_torch), trans_func(_yp, is_torch=is_torch)
    
    return loss_funcs[loss_func](_y, _yp)

def append_data(data_mse, data_mee, _data):
    data_mse.append(_data[0])
    data_mee.append(_data[1])

def eval_targets(model, dataset, outM):
    model.train(False)
    if hasattr(dataset, 's_dataset'):
        mk.magic_append(eval_single(model, dataset.s_dataset, 0, outM), 'data')
        for i,t_dataset in enumerate(dataset.t_datasets):
            mk.magic_append(eval_single(model, t_dataset, i+1, outM), 'data')
        data = np.row_stack(mk.magic_get('data'))
        data = np.hstack((data, np.mean(data, axis=1, keepdims=True), np.mean(data[:, 1:], axis=1, keepdims=True)))
    else:
        data = np.row_stack(eval_single(model, dataset, 0, outM))
    print(data)
    csvwrite(outM.get_filename('eval_result_loss'), data)

def eval_single(model, dataset, di, outM):
    eval_result = {}
    for i in range(len(dataset.index_map_list)):
        y_list = []; y_p_list = []
        init_psi = None
        for _,batch_data in enumerate(dataset.get_data_loader(100, seq_id = i)):
            _x, _y, _l, _i = data_to_device(batch_data)
            if init_psi is None:
                init_psi = _i
            _y_p = model.predict(_x, di)
            mse    = pred_loss(_y[..., :model.label_dim], _y_p, 'sparse', 'se', None, True)
            mseRho = pred_loss(_y[..., 0], _y_p[..., 0], 'sparse', 'se', None, True)
            msePsi = pred_loss(_y[..., 1], _y_p[..., 1], 'sparse', 'se', None, True)
            msePhi = pred_loss(_y[..., -1], _y_p[..., -1], 'sparse', 'se', None, True)
            mseAvg = torch.mean(torch.column_stack([mseRho, msePsi, msePhi]), dim=-1)
            mee    = pred_loss(_y[..., [0, -1]], _y_p[..., [0, -1]], 'sparse', 'ee', model.mee_trans_func, True)
            mk.magic_append([mse, mseAvg, mseRho, msePsi, msePhi, mee], 'errs')
            mk.magic_append([_y[:,-1,:], _y_p[:,-1,:], _l, _i], 'eval_result')
        _eval_result = mk.magic_get('eval_result', lambda x: t2n(torch.cat(x)))
        eval_result['%d_y'%i] = _eval_result[0]
        eval_result['%d_y_p'%i] = _eval_result[1]
        eval_result['%d_l'%i] = _eval_result[2]
        eval_result['%d_h'%i] = _eval_result[3]
        eval_result['%d_init_psi'%i] = t2n(init_psi)
    eval_result['seq_num'] = len(dataset.index_map_list)
    plot_prediction_label(eval_result, di, outM)
    save_h5(outM.get_filename('eval_result_trace_%d'%di, 'h5'), eval_result)
    errs = mk.magic_get('errs', lambda x: t2n(torch.cat(x)))
    errs_mean = [np.mean(err) for err in errs]
    errs_std = [np.std(err) for err in errs]
    return errs_mean + errs_std

def extract_features(model, dataset, outM):
    model.train(False)
    eval_result = {}
    if hasattr(dataset, 's_dataset'):
        eval_result[dataset.s_dataset.pose] = extract_single(model, dataset.s_dataset, 0, outM)
        for i,t_dataset in enumerate(dataset.t_datasets):
            eval_result[t_dataset.pose] = extract_single(model, t_dataset, i+1, outM)
    else:
        eval_result[dataset.pose] = extract_single(model, dataset, 0, outM)
    save_h5(outM.get_filename('extract_features', 'h5'), eval_result)

def extract_single(model, dataset, di, outM):
    z_p_list = []
    for i in range(len(dataset.index_map_list)):
        for _,batch_data in enumerate(dataset.get_data_loader(100, seq_id = i)):
            _x, _y, _l, _i = data_to_device(batch_data)
            _z_p = model.extract_feature(_x, di)
            z_p_list.append(_z_p)
    return t2n(torch.cat(z_p_list, dim=0)) 

def plot_prediction_label(eval_result, di, outM, start_ind = 200, end_ind = 2200):
    rows = ceil(eval_result['seq_num']/2)

    plt.figure(figsize=(2*10, rows*5))
    for i in range(eval_result['seq_num']):
        y = eval_result['%d_y'%i][start_ind:end_ind,:]
        y_p = eval_result['%d_y_p'%i][start_ind:end_ind,:]
        plt.subplot(rows, 2, i+1)
        plt.plot(range(y.shape[0]), y[:, 0], 'r')
        plt.plot(range(y.shape[0]), y_p[:, 0], 'b')
    plt.savefig(outM.get_filename('eval_result_trace_%d_length'%di, 'png'))

    plt.figure(figsize=(2*10, rows*5))
    for i in range(eval_result['seq_num']):
        plt.subplot(rows, 2, i+1)
        y = eval_result['%d_y'%i][start_ind:end_ind,:]
        y_p = eval_result['%d_y_p'%i][start_ind:end_ind,:]
        plt.plot(range(y.shape[0]), y[:, 1], 'r')
        plt.plot(range(y.shape[0]), y_p[:, 1], 'b')
    plt.savefig(outM.get_filename('eval_result_trace_%d_angleH'%di, 'png'))

    plt.figure(figsize=(2*10, rows*5))
    for i in range(eval_result['seq_num']):
        plt.subplot(rows, 2, i+1)
        y = eval_result['%d_y'%i][start_ind:end_ind,:]
        y_p = eval_result['%d_y_p'%i][start_ind:end_ind,:]
        plt.plot(range(y.shape[0]), y[:, -1], 'r')
        plt.plot(range(y.shape[0]), y_p[:, -1], 'b')
    plt.savefig(outM.get_filename('eval_result_trace_%d_angleW'%di, 'png'))