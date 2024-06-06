# -*- coding: UTF-8 -*-

import ipdb as pdb
import logging
log = logging.getLogger(__name__.split('.')[-1])

import numpy as np
np.set_printoptions(precision=4, suppress=True, formatter={'float_kind':'{:f}'.format})

'''
    根据 displacement_list 重建轨迹
    Input:
        displacement_list: np.array(N, 2 or 3) N个连续时间的相对位移
    Output:
        traj: np.array(N, 2 or 3): 重建的轨迹
'''
def recon_traj_with_preds(displacement_list):
    if displacement_list.shape[1] == 3:
        pos_now = np.array([0,0,0], dtype="float64")
    elif displacement_list.shape[1] == 2:
        pos_now = np.array([0,0], dtype="float64")

    traj = [pos_now.copy()]
    for dis in displacement_list:
        pos_now += dis
        traj.append(pos_now.copy())
    
    return np.array(traj)

def compute_absolute_trajectory_error(est, gt):
    """
    Args:
        est: estimated trajectory
        gt: ground truth trajectory. It must have the same shape as est.
    Return:
        Absolution trajectory error, which is the Root Mean Squared Error betweentwo trajectories.
    """
    return np.sqrt(np.mean((est - gt) ** 2)), np.std(np.sqrt((est - gt) ** 2))

def compute_relative_trajectory_error(est, gt, delta, max_delta=-1):
    """
    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """
    if max_delta == -1:
        max_delta = est.shape[0]
    deltas = np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[deltas[i]:] + gt[:-deltas[i]] - est[:-deltas[i]] - gt[deltas[i]:]
        rtes[i] = np.sqrt(np.mean(err ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(rtes), np.std(rtes)


def compute_ate_rte(est, gt, pred_per_min=60):
    """
    A convenient function to compute ATE and RTE. For sequences shorter than pred_per_min, it computes end sequence
    drift and scales the number accordingly.
    """
    ate, ate_std = compute_absolute_trajectory_error(est, gt)
    if est.shape[0] < pred_per_min:
        log.warn("est.shape[0] < pred_per_min")
        ratio = pred_per_min / est.shape[0]
        rte, rte_std = compute_relative_trajectory_error(est, gt, delta=est.shape[0] - 1)
        rte, rte_std = rte * ratio, rte_std * ratio
    else:
        rte, rte_std = compute_relative_trajectory_error(est, gt, delta=pred_per_min)

    return ate, rte, ate_std, rte_std
