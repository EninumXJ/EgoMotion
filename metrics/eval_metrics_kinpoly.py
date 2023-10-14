from collections import defaultdict
from metrics_util import *

import torch
import numpy as np

def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)

def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def compute_vel(joints):
    velocities = joints[1:] - joints[:-1]
    velocity_normed = np.linalg.norm(velocities, axis=2)
    return np.mean(velocity_normed, axis=1)

def compute_error_vel(joints_gt, joints_pred, vis = None):
    vel_gt = joints_gt[1:] - joints_gt[:-1] 
    vel_pred = joints_pred[1:] - joints_pred[:-1]
    normed = np.linalg.norm(vel_pred - vel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    return np.mean(normed[new_vis], axis=1)

def compute_metrics_for_smpl(gt_global_quat, gt_global_jpos, gt_floor_height, \
    pred_global_quat, pred_global_jpos, pred_floor_height):
    # T X J X 4, T X J X 3 

    res_dict = defaultdict(list)

    traj_pred = torch.cat((pred_global_jpos[:, 0, :], pred_global_quat[:, 0, :]), dim=-1).data.cpu().numpy() # T X 7 
    traj_gt = torch.cat((gt_global_jpos[:, 0, :], gt_global_quat[:, 0, :]), dim=-1).data.cpu().numpy() # T X 7 

    root_mat_pred = get_root_matrix(traj_pred)
    root_mat_gt = get_root_matrix(traj_gt)
    root_dist = get_frobenious_norm(root_mat_pred, root_mat_gt)
    root_rot_dist = get_frobenious_norm_rot_only(root_mat_pred, root_mat_gt)

    head_idx = 15 
    head_pose = torch.cat((pred_global_jpos[:, head_idx, :], pred_global_quat[:, head_idx, :]), dim=-1).data.cpu().numpy()
    head_pose_gt = torch.cat((gt_global_jpos[:, head_idx, :], gt_global_quat[:, head_idx, :]), dim=-1).data.cpu().numpy()

    head_mat_pred = get_root_matrix(head_pose)
    head_mat_gt = get_root_matrix(head_pose_gt)
    head_dist = get_frobenious_norm(head_mat_pred, head_mat_gt)
    head_rot_dist = get_frobenious_norm_rot_only(head_mat_pred, head_mat_gt)

    # Compute accl and accl err. 
    accels_pred = np.mean(compute_accel(pred_global_jpos.data.cpu().numpy())) * 1000
    accels_gt = np.mean(compute_accel(gt_global_jpos.data.cpu().numpy())) * 1000 

    accel_dist = np.mean(compute_error_accel(pred_global_jpos.data.cpu().numpy(), gt_global_jpos.data.cpu().numpy())) * 1000

    # Compute foot sliding error
    # pred_fs_metric = compute_foot_sliding_for_smpl(pred_global_jpos.data.cpu().numpy().copy(), pred_floor_height)
    # gt_fs_metric = compute_foot_sliding_for_smpl(gt_global_jpos.data.cpu().numpy().copy(), gt_floor_height)

    jpos_pred = pred_global_jpos - pred_global_jpos[:, 0:1] # zero out root
    jpos_gt =  gt_global_jpos - gt_global_jpos[:, 0:1] 
    jpos_pred = jpos_pred.data.cpu().numpy()
    jpos_gt = jpos_gt.data.cpu().numpy()
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis = 2).mean() * 1000

    # Add jpe for each joint 
    single_jpe = np.linalg.norm(jpos_pred - jpos_gt, axis = 2).mean(axis=0) * 1000 # J 
    
    # Remove joints 18, 19, 20, 21 
    mpjpe_wo_hand = single_jpe[:18].mean()

    # Jiaman: add root translation error 
    pred_root_trans = traj_pred[:, :3] # T X 3
    gt_root_trans = traj_gt[:, :3] # T X 3 
    root_trans_err = np.linalg.norm(pred_root_trans - gt_root_trans, axis = 1).mean() * 1000
    res_dict["root_trans_dist"].append(root_trans_err)

    # Add accl and accer 
    res_dict['accel_pred'] = accels_pred 
    res_dict['accel_gt'] = accels_gt 
    res_dict['accel_err'] = accel_dist 

    # Add foot sliding metric 
    # res_dict['pred_fs'] = pred_fs_metric 
    # res_dict['gt_fs'] = gt_fs_metric  

    pred_head_trans = head_pose[:, :3]
    gt_head_trans = head_pose_gt[:, :3] 
    head_trans_err = np.linalg.norm(pred_head_trans - gt_head_trans, axis = 1).mean() * 1000
    res_dict["head_trans_dist"].append(head_trans_err)

    res_dict["root_dist"].append(root_dist)
    res_dict["root_rot_dist"].append(root_rot_dist)
    res_dict["mpjpe"].append(mpjpe)
    res_dict["mpjpe_wo_hand"].append(mpjpe_wo_hand)
    res_dict["head_dist"].append(head_dist)
    res_dict["head_rot_dist"].append(head_rot_dist)
   
    res_dict['single_jpe'].append(single_jpe)
    for tmp_idx in range(single_jpe.shape[0]):
        res_dict['jpe_'+str(tmp_idx)].append(single_jpe[tmp_idx])

    res_dict = {k: np.mean(v) for k, v in res_dict.items()}
   
    return res_dict