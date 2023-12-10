import os
import shutil
import numpy as np
import argparse
import random
from time import time
from tqdm import tqdm
import wandb
import prettytable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pytorch3d.transforms as transforms 
from torch.utils.data import DataLoader, random_split
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
# from transformers import AutoImageProcessor, TimesformerForVideoClassification
from model.timesformer.models.vit import TimeSformer

from finetune import set_random_seed
from MotionBERT.lib.utils.learning import *
from MotionBERT.lib.utils.tools import *
from model.loss import *
from dataset.kinpoly_dataset import output2quat, output2matrix, KinPolyDataset, NewKinPolyDataset
from model.proj import Projector
from utils.data_utils.process_kinpoly_qpos2smpl import qpos2smpl, simpleqpos2smpl
from model.selformer import *

# os.environ["WANDB_API_KEY"] = "5513dc34ee28da741530ff8e9f9f3c13b9bc37ca"
# os.environ["WANDB_MODE"] = "offline"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/data/newhome/litianyi/logs/EgoMotion/", help="Path to the log file.")
    parser.add_argument("--config", type=str, default="config/DST_slam_train.yaml", help="Path to the config file.")
    # parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-backbone', '--backbone', default='Timesformer', type=str, help='backbone of EgoMotion')
    parser.add_argument('--resume', default='', type=str, help='backbone of EgoMotion')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-p', '--projector', choices=['linear', 'rnn', 'transformer'], type=str, help='projector type of EgoMotion')
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    opts = parser.parse_args()
    return opts

def evaluate_linear(args, model, proj, val_loader, feature_extractor=None, backbone=None):
    print('INFO: Testing')
    results_all = []; gt_all = []; losses_all = {}
    losses_all['total'] = AverageMeter()
    losses_all['loss_quat_v'] = AverageMeter()
    losses_all['loss_quat'] = AverageMeter()
    losses_all['loss_3d_pos'] = AverageMeter()
    losses_all['loss_3d_scale'] = AverageMeter()
    losses_all['loss_3d_velocity'] = AverageMeter()
    model.eval()
    proj.eval()
    num_joints = args.num_joints           
    with torch.no_grad():
        with tqdm(total=args.val_size) as pbar:
            for idx, (lrot_gt, img_clip, root, joint_rot, _) in enumerate(val_loader):
                N, C = img_clip.shape[:2]
                if torch.cuda.is_available():
                    lrot_gt = lrot_gt.cuda()[:,1:]  ## delete the first one(zero)
                    img_clip = img_clip.cuda()
                    root = root.cuda()
                    joint_rot = joint_rot.cuda()
                if args.use_depth:    
                    # predict depth
                    img_clip = img_clip.reshape(-1, 3, 224, 224)
                    with torch.no_grad():
                        depth = depth_estimate(img_clip*255, feature_extractor, backbone)  ### Normalize->[0,255]
                    depth = depth.reshape(N, C, 1, 224, 224)
                    # Predict 3D poses
                    inputs = torch.repeat_interleave(depth, 3, dim=2).permute(0, 2, 1, 3, 4)
                else:
                    inputs = img_clip.permute(0, 2, 1, 3, 4)
                embeddings = model(inputs)  # (N, F, J, C) = (32, 10, 23, 3)
                if args.rot_represent == 'quat4d':
                    output = proj(embeddings).reshape(N, -1, 23, 4)    # (N, T-1, 23, 4)
                    predicted_joint_quat_diff = output
                    # compute 3d pos
                    predicted_joint_quat = output2quat(predicted_joint_quat_diff, joint_rot)  # (N, T, 23, 4)
                    # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                    # output quat to euler
                    joint_euler_status = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(predicted_joint_quat), convention='ZYX')
                    root = root.reshape(-1, 7)
                    predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                    # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                    batch_gt = qpos2smpl(joint_rot.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                    batch_gt = batch_gt.reshape(N, -1, 22, 3)
                    predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
                elif args.rot_represent == 'rot6d':
                    output = proj(embeddings).reshape(N, -1, num_joints, 6)    # (N, T-1, num_joints, 6)
                    predicted_joint_rot6d = output    # (N, T-1, num_joints, 6)
                    predicted_joint_mat = transforms.rotation_6d_to_matrix(predicted_joint_rot6d)  # (N, T-1, num_joints, 3, 3)
                    root_quat_initial = root[:,0:1,None,3:]  # (N, 1, 1, 4)
                    predicted_root_rot_diff = transforms.matrix_to_quaternion(predicted_joint_mat[:,:,0:1,...])  # (N, T-1, 1, 4)
                    predicted_others_rot_diff = predicted_joint_mat[:,:,1:,...]  # (N, T-1, num_joints-1, 3, 3)
                    predicted_joint_matrix = output2matrix(predicted_others_rot_diff, joint_rot, mode='to-initial')  # (N, T, 23, 3, 3)
                    predicted_root_rot = output2quat(predicted_root_rot_diff, root_quat_initial, mode='to-initial')  # (N, T, 1, 4)
                    # output quat to euler
                    joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX')
                    root = root.reshape(-1, 7)
                    predicted_root_rot = predicted_root_rot.reshape(-1, 4)  # (N*T, 4)
                    predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, num_joints-1, 3), predicted_root_rot, root[:, 0:3])
                    # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                    batch_gt = qpos2smpl(joint_rot.reshape(-1, num_joints-1, 3), root[:, 3:], root[:, 0:3])
                    batch_gt = batch_gt.reshape(N, -1, 22, 3)
                    predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
                
                loss_quat_ = loss_quat(output, lrot_gt)
                loss_quat_v_ = loss_quat_v(output, lrot_gt)
                loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
                loss_total =    args.lambda_quat_error * loss_quat_  + \
                                args.lambda_quat_velocvity * loss_quat_v_ + \
                                args.lambda_pos_error * loss_3d_pos + \
                                args.lambda_pos_scale * loss_3d_scale + \
                                args.lambda_pos_velocity * loss_3d_velocity
                
                losses_all['loss_quat'].update(loss_quat_.item(), N)
                losses_all['loss_quat_v'].update(loss_quat_v_.item(), N)
                losses_all['loss_3d_pos'].update(loss_3d_pos.item(), N)
                losses_all['loss_3d_scale'].update(loss_3d_scale.item(), N)
                losses_all['loss_3d_velocity'].update(loss_3d_velocity.item(), N)
                losses_all['total'].update(loss_total.item(), N)
                
                results_all.append(predicted_3d_pos.cpu().numpy())
                gt_all.append(batch_gt.cpu().numpy())
                if idx % 10 == 0:
                    pbar.update(10)
                # if idx > args.val_size:
                #     break
    
    gt_all = np.concatenate(gt_all)
    results_all = np.concatenate(results_all)
    print("test clips: ", results_all.shape[0])
    num_test_clips = len(results_all)
    e1_all = np.zeros(num_test_clips)
    e2_all = np.zeros(num_test_clips)
    oc = np.zeros(num_test_clips)
    
    final_results_1 = []; final_results_2 = []
    for idx in range(num_test_clips):
        gt = gt_all[idx]
        pred = results_all[idx]
        # Root-relative Errors
        # pred = pred - pred[:,0:1,:]
        # gt = gt - gt[:,0:1,:]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        
        e1_all[idx] = np.mean(err1)
        e2_all[idx] = np.mean(err2)
        oc[idx] += 1
        final_results_1.append(e1_all[idx])
        final_results_2.append(e2_all[idx])
    # for idx in range(num_test_frames):
    #     if e1_all[idx] > 0:
    #         err1 = e1_all[idx] / oc[idx]
    #         err2 = e2_all[idx] / oc[idx]
    #         final_results_1.append(err1)
    #         final_results_2.append(err2)
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name']
    
    summary_table.add_row(['P1'])
    summary_table.add_row(['P2'])
    print(summary_table)
    e1 = np.mean(np.array(final_results_1)) * 1000
    e2 = np.mean(np.array(final_results_2)) * 1000
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')
    return e1, e2, losses_all

def evaluate_rnn(args, model, proj, val_loader, feature_extractor=None, backbone=None):
    print('INFO: Testing')
    results_all = []; gt_all = []; losses_all = {}
    losses_all['total'] = AverageMeter()
    losses_all['loss_quat_v'] = AverageMeter()
    losses_all['loss_quat'] = AverageMeter()
    losses_all['loss_3d_pos'] = AverageMeter()
    losses_all['loss_3d_scale'] = AverageMeter()
    losses_all['loss_3d_velocity'] = AverageMeter()
    model.eval()
    proj.eval()
    num_joints = args.num_joints           
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for idx, (lrot_gt, img_clip, root, joint_rot, _) in enumerate(val_loader):
                N, C = img_clip.shape[:2]
                if torch.cuda.is_available():
                    lrot_gt = lrot_gt.cuda()[:,1:]  ## delete the first one(zero)
                    img_clip = img_clip.cuda()
                    root = root.cuda()
                    joint_rot = joint_rot.cuda()
                if args.use_depth:    
                    # predict depth
                    img_clip = img_clip.reshape(-1, 3, 224, 224)
                    with torch.no_grad():
                        depth = depth_estimate(img_clip*255, feature_extractor, backbone)  ### Normalize->[0,255]
                    depth = depth.reshape(N, C, 1, 224, 224)
                    # Predict 3D poses
                    inputs = torch.repeat_interleave(depth, 3, dim=2).permute(0, 2, 1, 3, 4)
                else:
                    inputs = img_clip.permute(0, 2, 1, 3, 4)
                embeddings = model(inputs)  # (N, F, J, C) = (32, 10, 23, 3)
                embeddings = embeddings.reshape(N, C-1, args.dim_feat).permute(1, 0, 2)  # (7, N, 144)
                # print("embeddings shape: ", embeddings.shape)
                joint_rot_6d = transforms.matrix_to_rotation_6d(transforms.euler_angles_to_matrix(joint_rot, convention='ZYX'))[:, 0:1, ...]  # (N, 1, 23, 6)
                root_rot_6d = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(root[:,0:1,3:]))[:,:,None,:]  # (N, 1, 1, 6)
                joint_rot_6d = torch.cat([root_rot_6d, joint_rot_6d], dim=2)   # (N, 1, 24, 6)
                initial_state = joint_rot_6d.reshape(N, 1, -1).permute(1, 0, 2)  # (1, N, 24*6)
                h0 = torch.repeat_interleave(initial_state, args.rnn_hidden_layers, dim=0)  # (num_layers, N, 24*6)
                if args.rot_represent == 'quat4d':
                    output = proj(embeddings, h0)[0].permute(1, 0, 2).reshape(N, -1, num_joints, 4)    # (N, T-1, 24, 4)
                    predicted_joint_quat_diff = output
                    # compute 3d pos
                    predicted_joint_quat = output2quat(predicted_joint_quat_diff, joint_rot)  # (N, T, 23, 4)
                    # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                    # output quat to euler
                    joint_euler_status = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(predicted_joint_quat), convention='ZYX')
                    root = root.reshape(-1, 7)
                    predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                    # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                    batch_gt = qpos2smpl(joint_rot.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                    batch_gt = batch_gt.reshape(N, -1, 22, 3)
                    predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
                elif args.rot_represent == 'rot6d':
                    output = proj(embeddings, h0)[0].permute(1, 0, 2).reshape(N, -1, num_joints, 6)    # (N, T-1, 24, 6)
                    predicted_joint_rot6d = output    # (N, T-1, num_joints, 6)
                    predicted_joint_mat = transforms.rotation_6d_to_matrix(predicted_joint_rot6d)  # (N, T-1, num_joints, 3, 3)
                    root_quat_initial = root[:,0:1,None,3:]  # (N, 1, 1, 4)
                    predicted_root_rot_diff = transforms.matrix_to_quaternion(predicted_joint_mat[:,:,0:1,...])  # (N, T-1, 1, 4)
                    predicted_others_rot_diff = predicted_joint_mat[:,:,1:,...]  # (N, T-1, num_joints-1, 3, 3)
                    predicted_joint_matrix = output2matrix(predicted_others_rot_diff, joint_rot, mode='to-initial')  # (N, T, 23, 3, 3)
                    predicted_root_rot = output2quat(predicted_root_rot_diff, root_quat_initial, mode='to-initial')  # (N, T, 1, 4)
                    # output quat to euler
                    joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX')
                    if args.num_joints == 24:
                        root = root.reshape(-1, 7)
                        predicted_root_rot = predicted_root_rot.reshape(-1, 4)  # (N*T, 4)
                        predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, num_joints-1, 3), predicted_root_rot, root[:, 0:3])
                        # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                        batch_gt = qpos2smpl(joint_rot.reshape(-1, num_joints-1, 3), root[:, 3:], root[:, 0:3])
                    elif args.num_joints == 13:
                        predicted_3d_pos = simpleqpos2smpl(joint_euler_status, predicted_root_rot, root[:, :, 0:3])
                        # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                        batch_gt = simpleqpos2smpl(joint_rot, root[:, :, 3:], root[:, :, 0:3])
                    else:
                        raise TypeError('Unsupported human joints num!')
                    batch_gt = batch_gt.reshape(N, -1, 22, 3)
                    predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
                
                loss_quat_ = loss_quat(output, lrot_gt)
                loss_quat_v_ = loss_quat_v(output, lrot_gt)
                loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
                loss_total =    args.lambda_quat_error * loss_quat_  + \
                                args.lambda_quat_velocvity * loss_quat_v_ + \
                                args.lambda_pos_error * loss_3d_pos + \
                                args.lambda_pos_scale * loss_3d_scale + \
                                args.lambda_pos_velocity * loss_3d_velocity
                
                losses_all['loss_quat'].update(loss_quat_.item(), N)
                losses_all['loss_quat_v'].update(loss_quat_v_.item(), N)
                losses_all['loss_3d_pos'].update(loss_3d_pos.item(), N)
                losses_all['loss_3d_scale'].update(loss_3d_scale.item(), N)
                losses_all['loss_3d_velocity'].update(loss_3d_velocity.item(), N)
                losses_all['total'].update(loss_total.item(), N)
                
                results_all.append(predicted_3d_pos.cpu().numpy())
                gt_all.append(batch_gt.cpu().numpy())
                if idx % 10 == 0:
                    pbar.update(10)
                # if idx > args.val_size:
                #     break
    
    gt_all = np.concatenate(gt_all)
    results_all = np.concatenate(results_all)
    print("test clips: ", results_all.shape[0])
    num_test_clips = len(results_all)
    e1_all = np.zeros(num_test_clips)
    e2_all = np.zeros(num_test_clips)
    oc = np.zeros(num_test_clips)
    
    final_results_1 = []; final_results_2 = []
    for idx in range(num_test_clips):
        gt = gt_all[idx]
        pred = results_all[idx]
        # Root-relative Errors
        # pred = pred - pred[:,0:1,:]
        # gt = gt - gt[:,0:1,:]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        
        e1_all[idx] = np.mean(err1)
        e2_all[idx] = np.mean(err2)
        oc[idx] += 1
        final_results_1.append(e1_all[idx])
        final_results_2.append(e2_all[idx])
    # for idx in range(num_test_frames):
    #     if e1_all[idx] > 0:
    #         err1 = e1_all[idx] / oc[idx]
    #         err2 = e2_all[idx] / oc[idx]
    #         final_results_1.append(err1)
    #         final_results_2.append(err2)
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name']
    
    summary_table.add_row(['P1'])
    summary_table.add_row(['P2'])
    print(summary_table)
    e1 = np.mean(np.array(final_results_1)) * 1000
    e2 = np.mean(np.array(final_results_2)) * 1000
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')
    return e1, e2, losses_all

def evaluate_transformer(args, backbone, transformer, val_loader, feature_extractor=None, depth_backbone=None):
    print('INFO: Testing')
    results_all = []; gt_all = []; losses_all = {}
    losses_all['total'] = AverageMeter()
    losses_all['loss_quat_v'] = AverageMeter()
    losses_all['loss_quat'] = AverageMeter()
    losses_all['loss_3d_pos'] = AverageMeter()
    losses_all['loss_3d_scale'] = AverageMeter()
    losses_all['loss_3d_velocity'] = AverageMeter()
    transformer.eval()
    num_joints = args.num_joints           
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for idx, (lrot_gt, img_clip, root, joint_rot, _) in enumerate(val_loader):
                N, L, C = img_clip.shape[0:3]
                # print("input shape: ", batch_input.shape)
                if torch.cuda.is_available():
                    img_clip = img_clip.cuda() # B,15,8,3,224,224
                    lrot_gt = lrot_gt.cuda()  #B,15,13,6
                    joint_rot = joint_rot.cuda() # B,16,12,3
                    root = root.cuda()  # B,16,7
                if isinstance(transformer, torch.nn.DataParallel):
                    transformer = transformer.module
                if args.use_depth:    
                    # predict depth
                    img_clip = img_clip.reshape(-1, 3, 224, 224)
                    with torch.no_grad():
                        depth = depth_estimate(img_clip*255, feature_extractor, depth_backbone)  ### Normalize->[0,255]
                    depth = depth.reshape(N, C, 1, 224, 224)
                    # Predict 3D poses
                    inputs = torch.repeat_interleave(depth, 3, dim=2).permute(0, 2, 1, 3, 4)
                else:
                    inputs = img_clip.reshape(-1, C, 3, 224, 224).permute(0, 2, 1, 3, 4)
                feature = backbone(inputs).reshape(N, L, -1)  # (N, L, 256)
                src_mask = None
                joint_rot_6d = transforms.matrix_to_rotation_6d(transforms.euler_angles_to_matrix(joint_rot, convention='ZYX'))[:, 0:1, ...]  # (N, 1, 23, 6)
                root_rot_6d = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(root[:,0:1,3:]))[:,:,None,:]  # (N, 1, 1, 6)
                joint_rot_6d = torch.cat([root_rot_6d, joint_rot_6d], dim=2)   # (N, 1, 24, 6)
                
                if args.rot_represent == 'quat4d':
                    output = transformer(feature)[0].permute(1, 0, 2).reshape(N, -1, num_joints, 4)    # (N, T-1, 24, 4)
                    predicted_joint_quat_diff = output
                    # compute 3d pos
                    predicted_joint_quat = output2quat(predicted_joint_quat_diff, joint_rot)  # (N, T, 23, 4)
                    # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                    # output quat to euler
                    joint_euler_status = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(predicted_joint_quat), convention='ZYX')
                    root = root.reshape(-1, 7)
                    predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                    # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                    batch_gt = qpos2smpl(joint_rot.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                    batch_gt = batch_gt.reshape(N, -1, 22, 3)
                    predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
                elif args.rot_represent == 'rot6d':
                    memory = transformer.encode(feature, src_mask)    # (N, T-1, 24, 6)
                    ys = torch.zeros(N, 1, num_joints*6).type_as(joint_rot).cuda()
                    for i in range(L):
                        tgt_mask = build_attention_mask_batch(N, args.num_heads, ys.size(1)).cuda()
                        out = transformer.decode(
                            memory, src_mask, ys, tgt_mask)
                        pose = transformer.generator(out[:, -1, ...]).unsqueeze(1)
                        # pose shape: (1, pose_dim)->(1, 1, pose_dim)
                        ys = torch.cat([ys, pose], dim=1)
                    output = ys[:, 1:, ...].reshape(N, L, num_joints, 6)
                    predicted_joint_rot6d = output    # (N, T-1, num_joints, 6)
                    predicted_joint_mat = transforms.rotation_6d_to_matrix(predicted_joint_rot6d)  # (N, T-1, num_joints, 3, 3)
                    root_quat_initial = root[:,0:1,None,3:]  # (N, 1, 1, 4)
                    predicted_root_rot_diff = transforms.matrix_to_quaternion(predicted_joint_mat[:,:,0:1,...])  # (N, T-1, 1, 4)
                    predicted_others_rot_diff = predicted_joint_mat[:,:,1:,...]  # (N, T-1, num_joints-1, 3, 3)
                    predicted_joint_matrix = output2matrix(predicted_others_rot_diff, joint_rot, mode='to-initial')  # (N, T, 23, 3, 3)
                    predicted_root_rot = output2quat(predicted_root_rot_diff, root_quat_initial, mode='to-initial')  # (N, T, 1, 4)
                    # output quat to euler
                    joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX')
                    if args.num_joints == 24:
                        root = root.reshape(-1, 7)
                        predicted_root_rot = predicted_root_rot.reshape(-1, 4)  # (N*T, 4)
                        predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, num_joints-1, 3), predicted_root_rot, root[:, 0:3])
                        # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                        batch_gt = qpos2smpl(joint_rot.reshape(-1, num_joints-1, 3), root[:, 3:], root[:, 0:3])
                    elif args.num_joints == 13:
                        predicted_3d_pos = simpleqpos2smpl(joint_euler_status, predicted_root_rot, root[:, :, 0:3])
                        # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                        batch_gt = simpleqpos2smpl(joint_rot, root[:, :, 3:], root[:, :, 0:3])
                    else:
                        raise TypeError('Unsupported human joints num!')
                    batch_gt = batch_gt.reshape(N, -1, 22, 3)[:,1:,...]
                    predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)[:,1:,...]
                
                loss_quat_ = loss_quat(output, lrot_gt)
                loss_quat_v_ = loss_quat_v(output, lrot_gt)
                loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
                loss_total =    args.lambda_quat_error * loss_quat_  + \
                                args.lambda_quat_velocvity * loss_quat_v_ + \
                                args.lambda_pos_error * loss_3d_pos + \
                                args.lambda_pos_scale * loss_3d_scale + \
                                args.lambda_pos_velocity * loss_3d_velocity
                
                losses_all['loss_quat'].update(loss_quat_.item(), N)
                losses_all['loss_quat_v'].update(loss_quat_v_.item(), N)
                losses_all['loss_3d_pos'].update(loss_3d_pos.item(), N)
                losses_all['loss_3d_scale'].update(loss_3d_scale.item(), N)
                losses_all['loss_3d_velocity'].update(loss_3d_velocity.item(), N)
                losses_all['total'].update(loss_total.item(), N)
                
                results_all.append(predicted_3d_pos.cpu().numpy())
                gt_all.append(batch_gt.cpu().numpy())
                if idx % 10 == 0:
                    pbar.update(10)
                # if idx > args.val_size:
                #     break
    
    gt_all = np.concatenate(gt_all)
    results_all = np.concatenate(results_all)
    print("test clips: ", results_all.shape[0])
    num_test_clips = len(results_all)
    e1_all = np.zeros(num_test_clips)
    e2_all = np.zeros(num_test_clips)
    oc = np.zeros(num_test_clips)
    
    final_results_1 = []; final_results_2 = []
    for idx in range(num_test_clips):
        gt = gt_all[idx]
        pred = results_all[idx]
        # Root-relative Errors
        # pred = pred - pred[:,0:1,:]
        # gt = gt - gt[:,0:1,:]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        
        e1_all[idx] = np.mean(err1)
        e2_all[idx] = np.mean(err2)
        oc[idx] += 1
        final_results_1.append(e1_all[idx])
        final_results_2.append(e2_all[idx])
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name']
    
    summary_table.add_row(['P1'])
    summary_table.add_row(['P2'])
    print(summary_table)
    e1 = np.mean(np.array(final_results_1)) * 1000
    e2 = np.mean(np.array(final_results_2)) * 1000
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')
    return e1, e2, losses_all

def depth_estimate(image, feature_extractor, model):

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    # prediction = torch.nn.functional.interpolate(
    #     predicted_depth.unsqueeze(1),
    #     size=image.size[::-1],
    #     mode="bicubic",
    #     align_corners=False,
    # )
    # print("depth shape: ", predicted_depth.shape)
    MMAX, _ = torch.max(predicted_depth, dim=2, keepdim=True)
    MAX, _ = torch.max(MMAX, dim=1, keepdim=True)
    # print("max shape: ", MAX.shape)
    predicted_depth_norm = predicted_depth / (MAX + 0.0001)
    # visualize the prediction
    # output = predicted_depth.squeeze().cpu().numpy()
    # formatted = (output * 255 / np.max(output)).astype("uint8")
    # depth = Image.fromarray(formatted)
    return predicted_depth_norm

def train_epoch_linear(args, model, proj, train_loader, losses, optimizer, feature_extractor=None, backbone=None):
    model.train()
    proj.train()
    num_joints = args.num_joints
    with tqdm(total=len(train_loader)) as pbar:
        for idx, (lrot_gt, img_clip, root, joint_rot) in enumerate(train_loader):
            N, C = img_clip.shape[0:2]
            # print("input shape: ", batch_input.shape)
            if torch.cuda.is_available():
                img_clip = img_clip.cuda()
                lrot_gt = lrot_gt.cuda()[:, 1:]
                joint_rot = joint_rot.cuda()
                root = root.cuda()
            if args.use_depth:    
                # predict depth
                img_clip = img_clip.reshape(-1, 3, 224, 224)
                with torch.no_grad():
                    depth = depth_estimate(img_clip*255, feature_extractor, backbone)  ### Normalize->[0,255]
                depth = depth.reshape(N, C, 1, 224, 224)
                # Predict 3D poses
                inputs = torch.repeat_interleave(depth, 3, dim=2).permute(0, 2, 1, 3, 4)
            else:
                inputs = img_clip.permute(0, 2, 1, 3, 4)
            
            embeddings = model(inputs)
            
            if args.rot_represent == 'quat4d':
                predicted_joint_rot = proj(embeddings).reshape(N, -1, num_joints, 4)    # (N, T-1, 24, 4)
                
                predicted_root_rot = predicted_joint_rot[:,:,0:1,:]  # (N, T-1, 1, 4)
                root_quat_initial = root[:,:,3:]  # (N, T, 4)
                predicted_joint_quat = output2quat(predicted_joint_rot[:,:,1:,:], joint_rot, mode='to-initial')  # (N, T, 23, 4)
                predicted_root_quat = output2quat(predicted_root_rot, root_quat_initial, mode='to-initial')  # (N, T, 1, 4)
                # # output quat to euler
                joint_euler_status = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(predicted_joint_quat), convention='ZYX')
                root = root.reshape(-1, 7)
                predicted_root_quat = predicted_root_quat.reshape(-1, 4)
                predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, num_joints, 3), predicted_root_quat, root[:, 0:3])
                batch_gt = qpos2smpl(joint_rot.reshape(-1, num_joints, 3), root[:, 3:], root[:, 0:3])
                
                batch_gt = batch_gt.reshape(N, -1, 22, 3)
                predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
            elif args.rot_represent == 'rot6d':
                predicted_joint_rot6d = proj(embeddings).reshape(N, -1, num_joints, 6)    # (N, T-1, num_joints, 6)
                predicted_joint_mat = transforms.rotation_6d_to_matrix(predicted_joint_rot6d)  # (N, T-1, num_joints, 3, 3)
                root_quat_initial = root[:,0:1,None,3:]  # (N, 1, 1, 4)
                predicted_root_rot_diff = transforms.matrix_to_quaternion(predicted_joint_mat[:,:,0:1,...])  # (N, T-1, 1, 4)
                predicted_others_rot_diff = predicted_joint_mat[:,:,1:,...]  # (N, T-1, num_joints-1, 3, 3)
                predicted_joint_matrix = output2matrix(predicted_others_rot_diff, joint_rot, mode='to-initial')  # (N, T, 23, 3, 3)
                predicted_root_rot = output2quat(predicted_root_rot_diff, root_quat_initial, mode='to-initial')  # (N, T, 1, 4)
                # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                # output quat to euler
                joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX')
                root = root.reshape(-1, 7)  # (N*T, 4)
                predicted_root_rot = predicted_root_rot.reshape(-1, 4)
                predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, num_joints-1, 3), predicted_root_rot, root[:, 0:3])
                # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                batch_gt = qpos2smpl(joint_rot.reshape(-1, num_joints-1, 3), root[:, 3:], root[:, 0:3])
                batch_gt = batch_gt.reshape(N, -1, 22, 3)
                predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
           
            optimizer.zero_grad()
            loss_quat_ = loss_quat(predicted_joint_rot6d, lrot_gt)
            loss_quat_v_ = loss_quat_v(predicted_joint_rot6d, lrot_gt)
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_total =    args.lambda_quat_error * loss_quat_  + \
                            args.lambda_quat_velocvity * loss_quat_v_ + \
                            args.lambda_pos_error * loss_3d_pos + \
                            args.lambda_pos_scale * loss_3d_scale + \
                            args.lambda_pos_velocity * loss_3d_velocity

            losses['loss_quat'].update(loss_quat_.item(), N)
            losses['loss_quat_v'].update(loss_quat_v_.item(), N)
            losses['loss_3d_pos'].update(loss_3d_pos.item(), N)
            losses['loss_3d_scale'].update(loss_3d_scale.item(), N)
            losses['loss_3d_velocity'].update(loss_3d_velocity.item(), N)
            losses['total'].update(loss_total.item(), N)
            print("loss_total: ", loss_total.item())
            loss_total.backward()
            optimizer.step()
            # evaluate(model_pos, twin_net, proj, val_loader)
            if idx % 10 == 0:
                pbar.update(10)

def train_epoch_rnn(args, model, proj, train_loader, losses, optimizer, feature_extractor=None, backbone=None):
    model.train()
    proj.train()
    num_joints = args.num_joints
    with tqdm(total=len(train_loader)) as pbar:
        for idx, (lrot_gt, img_clip, root, joint_rot) in enumerate(train_loader):
            N, C = img_clip.shape[0:2]
            # print("input shape: ", batch_input.shape)
            if torch.cuda.is_available():
                img_clip = img_clip.cuda()
                lrot_gt = lrot_gt.cuda()[:, 1:]
                joint_rot = joint_rot.cuda()
                root = root.cuda()
            if args.use_depth:    
                # predict depth
                img_clip = img_clip.reshape(-1, 3, 224, 224)
                with torch.no_grad():
                    depth = depth_estimate(img_clip*255, feature_extractor, backbone)  ### Normalize->[0,255]
                depth = depth.reshape(N, C, 1, 224, 224)
                # Predict 3D poses
                inputs = torch.repeat_interleave(depth, 3, dim=2).permute(0, 2, 1, 3, 4)
            else:
                inputs = img_clip.permute(0, 2, 1, 3, 4)
            
            embeddings = model(inputs)  # (N, 7*224)
            embeddings = embeddings.reshape(N, C-1, args.dim_feat).permute(1, 0, 2)  # (7, N, 144)
            # print("embeddings shape: ", embeddings.shape)
            joint_rot_6d = transforms.matrix_to_rotation_6d(transforms.euler_angles_to_matrix(joint_rot, convention='ZYX'))[:, 0:1, ...]  # (N, 1, 23, 6)
            root_rot_6d = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(root[:,0:1,3:]))[:,:,None,:]  # (N, 1, 1, 6)
            joint_rot_6d = torch.cat([root_rot_6d, joint_rot_6d], dim=2)   # (N, 1, 24, 6)
            initial_state = joint_rot_6d.reshape(N, 1, -1).permute(1, 0, 2)  # (1, N, 24*6)
            h0 = torch.repeat_interleave(initial_state, args.rnn_hidden_layers, dim=0)  # (num_layers, N, 24*6)
            if args.rot_represent == 'quat4d':
                predicted_joint_rot = proj(embeddings, h0)[0].permute(1, 0, 2).reshape(N, -1, num_joints, 4)    # (N, T-1, 24, 4)
                
                predicted_root_rot = predicted_joint_rot[:,:,0:1,:]  # (N, T-1, 1, 4)
                root_quat_initial = root[:,:,3:]  # (N, T, 4)
                predicted_joint_quat = output2quat(predicted_joint_rot[:,:,1:,:], joint_rot, mode='to-initial')  # (N, T, 23, 4)
                predicted_root_quat = output2quat(predicted_root_rot, root_quat_initial, mode='to-initial')  # (N, T, 1, 4)
                # # output quat to euler
                joint_euler_status = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(predicted_joint_quat), convention='ZYX')
                root = root.reshape(-1, 7)
                predicted_root_quat = predicted_root_quat.reshape(-1, 4)
                predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, num_joints, 3), predicted_root_quat, root[:, 0:3])
                batch_gt = qpos2smpl(joint_rot.reshape(-1, num_joints, 3), root[:, 3:], root[:, 0:3])
                
                batch_gt = batch_gt.reshape(N, -1, 22, 3)
                predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
            elif args.rot_represent == 'rot6d':
                predicted_joint_rot6d = proj(embeddings, h0)[0].permute(1, 0, 2).reshape(N, C-1, num_joints, 6)    # (N, T-1, num_joints, 6)
                predicted_joint_mat = transforms.rotation_6d_to_matrix(predicted_joint_rot6d)  # (N, T-1, num_joints, 3, 3)
                root_quat_initial = root[:,0:1,None,3:]  # (N, 1, 1, 4)
                predicted_root_rot_diff = transforms.matrix_to_quaternion(predicted_joint_mat[:,:,0:1,...])  # (N, T-1, 1, 4)
                predicted_others_rot_diff = predicted_joint_mat[:,:,1:,...]  # (N, T-1, num_joints-1, 3, 3)
                predicted_joint_matrix = output2matrix(predicted_others_rot_diff, joint_rot, mode='to-initial')  # (N, T, 23, 3, 3)
                predicted_root_rot = output2quat(predicted_root_rot_diff, root_quat_initial, mode='to-initial')  # (N, T, 1, 4)
                # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                # output quat to euler
                joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX')
                if args.num_joints == 24:
                    root = root.reshape(-1, 7)  # (N*T, 7)
                    predicted_root_rot = predicted_root_rot.reshape(-1, 4)
                    predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, num_joints-1, 3), predicted_root_rot, root[:, 0:3])
                    # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                    batch_gt = qpos2smpl(joint_rot.reshape(-1, num_joints-1, 3), root[:, 3:], root[:, 0:3])
                elif args.num_joints == 13:
                    predicted_3d_pos = simpleqpos2smpl(joint_euler_status, predicted_root_rot, root[:, :, 0:3])
                    # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                    batch_gt = simpleqpos2smpl(joint_rot, root[:, :, 3:], root[:, :, 0:3])
                else:
                    raise TypeError('Unsupported human joints num!')
                batch_gt = batch_gt.reshape(N, -1, 22, 3)
                predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
           
            optimizer.zero_grad()
            loss_quat_ = loss_quat(predicted_joint_rot6d, lrot_gt)
            loss_quat_v_ = loss_quat_v(predicted_joint_rot6d, lrot_gt)
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_total =    args.lambda_quat_error * loss_quat_  + \
                            args.lambda_quat_velocvity * loss_quat_v_ + \
                            args.lambda_pos_error * loss_3d_pos + \
                            args.lambda_pos_scale * loss_3d_scale + \
                            args.lambda_pos_velocity * loss_3d_velocity

            losses['loss_quat'].update(loss_quat_.item(), N)
            losses['loss_quat_v'].update(loss_quat_v_.item(), N)
            losses['loss_3d_pos'].update(loss_3d_pos.item(), N)
            losses['loss_3d_scale'].update(loss_3d_scale.item(), N)
            losses['loss_3d_velocity'].update(loss_3d_velocity.item(), N)
            losses['total'].update(loss_total.item(), N)
            print("loss_total: ", loss_total.item())
            loss_total.backward()
            optimizer.step()
            # evaluate(model_pos, twin_net, proj, val_loader)
            if idx % 10 == 0:
                pbar.update(10)

def train_epoch_transformer(args, backbone, transformer, train_loader, losses, optimizer, feature_extractor=None, depth_backbone=None):
    transformer.train()
    num_joints = args.num_joints
    with tqdm(total=len(train_loader)) as pbar:
        for idx, (lrot_gt, img_clip, root, joint_rot, tgt_mask) in enumerate(train_loader):
            N, L, C = img_clip.shape[0:3]
            # print("clip length: ", L)
            # print("input shape: ", batch_input.shape)
            if torch.cuda.is_available():
                img_clip = img_clip.cuda() # B,15,8,3,224,224
                lrot_gt = lrot_gt.cuda()  #B,15,13,6
                joint_rot = joint_rot.cuda() # B,16,12,3
                root = root.cuda()  # B,16,7
                tgt_mask = tgt_mask.cuda()
            if isinstance(transformer, torch.nn.DataParallel):
                transformer = transformer.module
            if args.use_depth:    
                # predict depth
                img_clip = img_clip.reshape(-1, 3, 224, 224)
                with torch.no_grad():
                    depth = depth_estimate(img_clip*255, feature_extractor, backbone)  ### Normalize->[0,255]
                depth = depth.reshape(N, C, 1, 224, 224)
                # Predict 3D poses
                inputs = torch.repeat_interleave(depth, 3, dim=2).permute(0, 2, 1, 3, 4).contiguous()
            else:
                inputs = img_clip.reshape(-1, C, 3, 224, 224).permute(0, 2, 1, 3, 4).contiguous()  # (N*L, 3, C, 224, 224)
            
            optimizer.zero_grad()
            feature = backbone(inputs).reshape(N, L, -1)  # (N, L, 256)
            src_mask = None
            tgt_mask = tgt_mask[:,None,...]  # (N, 1, L, L)
            tgt_mask = torch.repeat_interleave(tgt_mask, args.num_heads, dim=1).reshape(-1, L, L)
            joint_rot_6d = transforms.matrix_to_rotation_6d(transforms.euler_angles_to_matrix(joint_rot, convention='ZYX'))[:, 0:1, ...]  # (N, 1, 12, 6)
            root_rot_6d = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(root[:,0:1,3:]))[:,:,None,:]  # (N, 1, 1, 6)
            joint_rot_6d = torch.cat([root_rot_6d, joint_rot_6d], dim=2)   # (N, 1, 13, 6)
            target = lrot_gt.reshape(N, L, -1)   # (N, L, 13*6)
            zeros_flag = torch.zeros_like(target[:,0:1,:])
            target = torch.cat([zeros_flag, target], dim=1)
            trg = target[:,:-1,...]
            # print("trg shape: ", trg.shape)
            trg_y = target[:,1:,...]
            if args.rot_represent == 'quat4d':
                predicted_joint_rot = transformer.forward(embeddings)[0].permute(1, 0, 2).reshape(N, -1, num_joints, 4)    # (N, T-1, 24, 4)
                
                predicted_root_rot = predicted_joint_rot[:,:,0:1,:]  # (N, T-1, 1, 4)
                root_quat_initial = root[:,:,3:]  # (N, T, 4)
                predicted_joint_quat = output2quat(predicted_joint_rot[:,:,1:,:], joint_rot, mode='to-initial')  # (N, T, 23, 4)
                predicted_root_quat = output2quat(predicted_root_rot, root_quat_initial, mode='to-initial')  # (N, T, 1, 4)
                # # output quat to euler
                joint_euler_status = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(predicted_joint_quat), convention='ZYX')
                root = root.reshape(-1, 7)
                predicted_root_quat = predicted_root_quat.reshape(-1, 4)
                predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, num_joints, 3), predicted_root_quat, root[:, 0:3])
                batch_gt = qpos2smpl(joint_rot.reshape(-1, num_joints, 3), root[:, 3:], root[:, 0:3])
                
                batch_gt = batch_gt.reshape(N, -1, 22, 3)
                predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
            elif args.rot_represent == 'rot6d':
                output = transformer.forward(feature, trg, src_mask, tgt_mask)
                predicted_joint_rot6d = transformer.generator(output).reshape(N, L, num_joints, 6)
                predicted_joint_mat = transforms.rotation_6d_to_matrix(predicted_joint_rot6d)  # (N, L, num_joints, 3, 3)
                root_quat_initial = root[:,0:1,None,3:]  # (N, 1, 1, 4)
                predicted_root_rot_diff = transforms.matrix_to_quaternion(predicted_joint_mat[:,:,0:1,...])  # (N, T, 1, 4)
                predicted_others_rot_diff = predicted_joint_mat[:,:,1:,...]  # (N, T, num_joints-1, 3, 3)
                predicted_joint_matrix = output2matrix(predicted_others_rot_diff, joint_rot, mode='to-initial')  # (N, T+1, 23, 3, 3)
                predicted_root_rot = output2quat(predicted_root_rot_diff, root_quat_initial, mode='to-initial')  # (N, T+1, 1, 4)
                # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                # output quat to euler
                joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX')
                if args.num_joints == 24:
                    root = root.reshape(-1, 7)  # (N*T, 7)
                    predicted_root_rot = predicted_root_rot.reshape(-1, 4)
                    predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, num_joints-1, 3), predicted_root_rot, root[:, 0:3])
                    # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                    batch_gt = qpos2smpl(joint_rot.reshape(-1, num_joints-1, 3), root[:, 3:], root[:, 0:3])
                elif args.num_joints == 13:
                    predicted_3d_pos = simpleqpos2smpl(joint_euler_status, predicted_root_rot, root[:, :, 0:3])
                    batch_gt = simpleqpos2smpl(joint_rot, root[:, :, 3:], root[:, :, 0:3])
                else:
                    raise TypeError('Unsupported human joints num!')
                batch_gt = batch_gt.reshape(N, -1, 22, 3)[:,1:,...]
                predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)[:,1:,...]
            loss_quat_ = loss_quat(predicted_joint_rot6d, lrot_gt)
            loss_quat_v_ = loss_quat_v(predicted_joint_rot6d, lrot_gt)
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_total =    args.lambda_quat_error * loss_quat_  + \
                            args.lambda_quat_velocvity * loss_quat_v_ + \
                            args.lambda_pos_error * loss_3d_pos + \
                            args.lambda_pos_scale * loss_3d_scale + \
                            args.lambda_pos_velocity * loss_3d_velocity

            losses['loss_quat'].update(loss_quat_.item(), N)
            losses['loss_quat_v'].update(loss_quat_v_.item(), N)
            losses['loss_3d_pos'].update(loss_3d_pos.item(), N)
            losses['loss_3d_scale'].update(loss_3d_scale.item(), N)
            losses['loss_3d_velocity'].update(loss_3d_velocity.item(), N)
            losses['total'].update(loss_total.item(), N)
            print("loss_total: ", loss_total.item())
            loss_total.backward()
            optimizer.step()
            # evaluate(model_pos, twin_net, proj, val_loader)
            if idx % 10 == 0:
                pbar.update(10)

def train(args, opts, model, proj, train_data, test_data, checkpoint=None, feature_extractor=None, backbone=None):
    st = 0
    min_loss = 100000
    if not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW([{"params": model.parameters()}, {"params": proj.parameters()}],
                                    lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']
    
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            train_dataset, _ = random_split(train_data, [args.train_size, len(train_data) - args.train_size])
            test_dataset, _ = random_split(test_data, [args.val_size, len(test_data) - args.val_size])
            val_dataset = test_dataset
            train_loader = DataLoader(train_dataset,
                                      batch_size=args.batch_size, shuffle=True, drop_last=True,
                                      num_workers=4, pin_memory=False)
            val_loader = DataLoader(val_dataset,
                                      batch_size=args.batch_size, shuffle=True, drop_last=True,
                                      num_workers=4, pin_memory=False)
            start_time = time()
            losses = {}
            losses['loss_quat'] = AverageMeter()
            losses['loss_quat_v'] = AverageMeter()
            losses['loss_3d_pos'] = AverageMeter()
            losses['loss_3d_scale'] = AverageMeter()
            losses['loss_3d_velocity'] = AverageMeter()
            losses['total'] = AverageMeter()
            
            if opts.projector == 'linear': 
                train_epoch_linear(args, model=model, proj=proj, train_loader=train_loader, losses=losses, optimizer=optimizer,
                            feature_extractor=feature_extractor, backbone=backbone)
            elif opts.projector == 'rnn':
                train_epoch_rnn(args, model=model, proj=proj, train_loader=train_loader, losses=losses, optimizer=optimizer,
                            feature_extractor=feature_extractor, backbone=backbone)
            else:
                raise ValueError("Unsupported projector.")
            
            elapsed = (time() - start_time) / 60    

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                if opts.projector == 'linear':
                    e1, e2, losses_test = evaluate_linear(args, model=model, proj=proj, val_loader=val_loader,
                                                        feature_extractor=feature_extractor, backbone=backbone)
                elif opts.projector == 'rnn':
                    e1, e2, losses_test = evaluate_rnn(args, model=model, proj=proj, val_loader=val_loader,
                                                        feature_extractor=feature_extractor, backbone=backbone)
                else:
                    raise ValueError("Unsupported projector.")
                # print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                #     epoch + 1,
                #     elapsed,
                #     lr,
                #     losses['3d_pos'].avg,
                #     e1, e2))
                # ---------------------------------------------------------------
                # wandb log
                if opts.use_wandb:
                    wandb.log({'epoch': epoch+1, 'Error P1': e1})
                    wandb.log({'epoch': epoch+1, 'Error P2': e2})
                    wandb.log({'epoch': epoch+1, 'loss_quat': losses['loss_quat'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_quat_v': losses['loss_quat_v'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_3d_pos': losses['loss_3d_pos'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_3d_scale': losses['loss_3d_scale'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_3d_velocity': losses['loss_3d_velocity'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_total_train': losses['total'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_total_test': losses_test['total'].avg})
            # ------------------------------------------------------------------- 
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # -------------------------------------------------------------------
            # Save checkpoints
            chk_dir = os.path.join(opts.log_dir, args.experiment_name, "checkpoints")
            if not os.path.exists(chk_dir):
                os.makedirs(chk_dir)
            chk_path = os.path.join(chk_dir, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(chk_dir, 'latest_epoch.pth')
            chk_path_best = os.path.join(chk_dir, 'best_epoch.pth')
            models = {'TimesFormer': model, 'proj': proj}
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, models, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, models, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, models, min_loss)

def main(args, opts):
    print(args)
    # ---------------------------------------------------------------------------
    # Dataset
    print('Loading dataset...')
    img_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224)),
                    torchvision.transforms.ToTensor()])
   
    train_data = KinPolyDataset(dataset_path=args.data_root,
                        config_path=args.config,
                        image_tmpl=args.image_tmpl,
                        transform=img_transforms,
                        clip_length=args.clip_len,
                        mode='train',
                        use_slam=True,
                        rot=args.rot_represent,
                        if_sample=args.if_sample,
                        num_of_keypoints=args.num_joints)
    test_data = KinPolyDataset(dataset_path=args.data_root,
                        config_path=args.config,
                        image_tmpl=args.image_tmpl,
                        transform=img_transforms,
                        clip_length=args.clip_len,
                        mode='test',
                        use_slam=True,
                        rot=args.rot_represent,
                        if_sample=args.if_sample,
                        num_of_keypoints=args.num_joints)

    # train_dataset, _ = random_split(train_data, [args.train_size, len(train_data) - args.train_size])
    # test_dataset, _ = random_split(test_data, [args.val_size, len(test_data) - args.val_size])
    # val_dataset = test_dataset
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True,
    #                           num_workers=4, pin_memory=False)
    # val_loader = DataLoader(val_dataset,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True,
    #                           num_workers=4, pin_memory=False)
    # ---------------------------------------------------------------------------
    # wandb Log
    run_dir = Path("/data/newhome/litianyi/logs/EgoMotion/" + args.experiment_name + "/wandb/")
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    if opts.use_wandb:
        wandb.init(
            project="Ego-Pose-Estimation(Kinpoly)",
            name=args.experiment_name,
            job_type="training",
            # hyperparameters
            config={
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.learning_rate,
                "clip_length": args.clip_len,
                "frames": args.clip_len if not args.if_sample else int(args.clip_len / 2),
                "joints num": args.num_joints,
                "pose_represent": "joint_rot--" + args.rot_represent
            },
            dir=str(run_dir),
        )
    # --------------------------------------------------------------------------
    # Model
    num_joints = args.num_joints
    num_frames = int(args.clip_len / 2) if args.if_sample else args.clip_len
    model = TimeSformer(img_size=224, num_classes=args.dim_feat*(num_frames-1), num_frames=args.clip_len, attention_type='divided_space_time',
                        pretrained_model="/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth")
    
    # projector
    if args.rot_represent == 'rot6d':
        if opts.projector == 'linear':
            proj = Projector(args.dim_feat*(num_frames-1), num_joints*6*(num_frames-1), 512, 1024)
        elif opts.projector == 'rnn':
            proj = nn.RNN(args.dim_feat, num_joints*6, args.rnn_hidden_layers)   # hidden_layers: 4
    elif args.rot_represent == 'quat4d':
        if opts.projector == 'linear':
            proj = Projector(args.dim_feat*(num_frames-1), num_joints*4*(num_frames-1), 512)
        elif opts.projector == 'rnn':
            proj = nn.RNN(args.dim_feat, num_joints*4, args.rnn_hidden_layers)
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    ### init depeth estimator
    if args.use_depth:
        feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
        backbone = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
        if torch.cuda.is_available():
            # feature_extractor = nn.DataParallel(feature_extractor)
            # feature_extractor = feature_extractor.cuda()
            backbone = nn.DataParallel(backbone)
            backbone = backbone.cuda()
    else:
        feature_extractor = None
        backbone = None

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        proj = nn.DataParallel(proj)
        proj = proj.cuda()

    checkpoint = None
    # ---------------------------------------------------------------------------
    # TODO resume from checkpoint
    if opts.resume:
        chk_filename = opts.resume
        print("Loading resume", chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['backbone'], strict=True)
        proj.load_state_dict(checkpoint['proj'], strict=True)
    #----------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # copy config to log dir
    config_dir = os.path.join(opts.log_dir, args.experiment_name, "config")
    opts_dir = os.path.join(opts.log_dir, args.experiment_name, "opts")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    if not os.path.exists(opts_dir):
        os.makedirs(opts_dir)
    srcfile = opts.config  
    dstfile = os.path.join(config_dir, 'configs.yaml')
    shutil.copyfile(srcfile, dstfile)
    setting_file = os.path.join(opts_dir, 'setting.txt')
    optsDict = opts.__dict__
    with open(setting_file, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in optsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    #----------------------------------------------------------------------------
    if not opts.evaluate:
        train(args, opts, model, proj, train_data, test_data, checkpoint, feature_extractor, backbone)
    else:
        e1, e2, results_all = evaluate(args, model, proj, test_data, feature_extractor, backbone) 

def save_checkpoint(ckpt_path, epoch, lr, optimizer, model, min_loss):
    print('Saving model checkpoint to', ckpt_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'backbone': model['backbone'].state_dict(),
        'transformer': model['transformer'].state_dict(),
        'min_loss' : min_loss
    }, ckpt_path)

def NewMain(args, opts):
    print(args)
    # ---------------------------------------------------------------------------
    # Dataset
    print('Loading dataset...')
    img_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224)),
                    torchvision.transforms.ToTensor()])
   
    train_data = NewKinPolyDataset(dataset_path=args.data_root,
                        config_path=args.config,
                        image_tmpl=args.image_tmpl,
                        transform=img_transforms,
                        clip_length=args.clip_len,
                        mode='train',
                        use_slam=True,
                        rot=args.rot_represent,
                        num_of_keypoints=args.num_joints)
    test_data = NewKinPolyDataset(dataset_path=args.data_root,
                        config_path=args.config,
                        image_tmpl=args.image_tmpl,
                        transform=img_transforms,
                        clip_length=args.clip_len,
                        mode='test',
                        use_slam=True,
                        rot=args.rot_represent,
                        num_of_keypoints=args.num_joints)

    # ---------------------------------------------------------------------------
    # wandb Log
    run_dir = Path("/data/newhome/litianyi/logs/EgoMotion/" + args.experiment_name + "/wandb/")
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    if opts.use_wandb:
        wandb.init(
            project="Ego-Pose-Estimation(Kinpoly)",
            name=args.experiment_name,
            job_type="training",
            # hyperparameters
            config={
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.learning_rate,
                "context_length": args.context_length,
                "frames": 8,
                "joints num": args.num_joints,
                "pose_represent": "joint_rot--" + args.rot_represent
            },
            dir=str(run_dir),
        )
    # --------------------------------------------------------------------------
    # Model
    num_joints = args.num_joints
    num_frames = 8

    backbone = TimeSformer(img_size=224, num_classes=args.dim_feat, num_frames=8, attention_type='divided_space_time',
                        pretrained_model="/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth")
    # Transformer
    if args.rot_represent == 'rot6d':
        transformer = make_model(src_feat=args.dim_feat, tgt_feat=args.num_joints*6,
                                layers=args.depth, d_model=args.dim_model, n_head=args.num_heads,
                                  dropout=args.dropout)
    if args.rot_represent == 'quat4d':
        transformer = make_model(src_feat=args.dim_feat, tgt_feat=args.num_joints*4,
                                layers=args.depth, d_model=args.dim_model, n_head=args.num_heads,
                                  dropout=args.dropout)
    model_params = 0
    for parameter in transformer.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    ### init depeth estimator
    if args.use_depth:
        feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
        depth_backbone = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
        if torch.cuda.is_available():
            # feature_extractor = nn.DataParallel(feature_extractor)
            # feature_extractor = feature_extractor.cuda()
            depth_backbone = nn.DataParallel(depth_backbone)
            depth_backbone = depth_backbone.cuda()
    else:
        feature_extractor = None
        depth_backbone = None

    if torch.cuda.is_available():
        transformer = nn.DataParallel(transformer)
        transformer = transformer.cuda()
        backbone = nn.DataParallel(backbone)
        backbone = backbone.cuda()

    checkpoint = None
    # ---------------------------------------------------------------------------
    # TODO resume from checkpoint
    if opts.resume:
        chk_filename = opts.resume
        print("Loading resume", chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        transformer.load_state_dict(checkpoint['transformer'], strict=True)
        backbone.load_state_dict(checkpoint['backbone'], strict=True)
    #----------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # copy config to log dir
    config_dir = os.path.join(opts.log_dir, args.experiment_name, "config")
    opts_dir = os.path.join(opts.log_dir, args.experiment_name, "opts")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    if not os.path.exists(opts_dir):
        os.makedirs(opts_dir)
    srcfile = opts.config  
    dstfile = os.path.join(config_dir, 'configs.yaml')
    shutil.copyfile(srcfile, dstfile)
    setting_file = os.path.join(opts_dir, 'setting.txt')
    optsDict = opts.__dict__
    with open(setting_file, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in optsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    #----------------------------------------------------------------------------
    if not opts.evaluate:
        NewTrain(args, opts, backbone, transformer, train_data, test_data, checkpoint)

def NewTrain(args, opts, backbone, transformer, train_data, test_data, checkpoint=None, feature_extractor=None, depth_backbone=None):
    st = 0
    min_loss = 100000
    if not opts.evaluate:
        lr = args.learning_rate
        # Freeze backbone
        for name, parameter in backbone.named_parameters():
            parameter.requires_grad = False
            if "head" in name:
                parameter.requires_grad = True
            if "blocks.11.mlp" in name:
                parameter.requires_grad = True

        optimizer = optim.AdamW([{"params": filter(lambda p : p.requires_grad, backbone.parameters()), 'lr':0.01}, 
                                 {"params": transformer.parameters()}],
                                    lr=lr, weight_decay=args.weight_decay)
        
        lr_decay = args.lr_decay
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']

        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            train_dataset, _ = random_split(train_data, [args.train_size, len(train_data) - args.train_size])
            test_dataset, _ = random_split(test_data, [args.val_size, len(test_data) - args.val_size])
            val_dataset = test_dataset
            train_loader = DataLoader(train_dataset,
                                      batch_size=args.batch_size, shuffle=True, drop_last=True,
                                      num_workers=4, pin_memory=False)
            val_loader = DataLoader(val_dataset,
                                      batch_size=args.batch_size, shuffle=True, drop_last=True,
                                      num_workers=4, pin_memory=False)
            start_time = time()
            losses = {}
            losses['loss_quat'] = AverageMeter()
            losses['loss_quat_v'] = AverageMeter()
            losses['loss_3d_pos'] = AverageMeter()
            losses['loss_3d_scale'] = AverageMeter()
            losses['loss_3d_velocity'] = AverageMeter()
            losses['total'] = AverageMeter()
            
            train_epoch_transformer(args, backbone, transformer, train_loader, losses, optimizer)
            
            elapsed = (time() - start_time) / 60    

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                e1, e2, losses_test = evaluate_transformer(args, backbone, transformer, val_loader)
                
                # ---------------------------------------------------------------
                # wandb log
                if opts.use_wandb:
                    wandb.log({'epoch': epoch+1, 'Error P1': e1})
                    wandb.log({'epoch': epoch+1, 'Error P2': e2})
                    wandb.log({'epoch': epoch+1, 'loss_quat': losses['loss_quat'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_quat_v': losses['loss_quat_v'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_3d_pos': losses['loss_3d_pos'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_3d_scale': losses['loss_3d_scale'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_3d_velocity': losses['loss_3d_velocity'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_total_train': losses['total'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_total_test': losses_test['total'].avg})
            # ------------------------------------------------------------------- 
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # -------------------------------------------------------------------
            # Save checkpoints
            chk_dir = os.path.join(opts.log_dir, args.experiment_name, "checkpoints")
            if not os.path.exists(chk_dir):
                os.makedirs(chk_dir)
            
            chk_path_latest = os.path.join(chk_dir, 'latest_epoch_model.pth')
            chk_path_best = os.path.join(chk_dir, 'best_epoch_model.pth')
            models = {'transformer': transformer, 'backbone': backbone}
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, models, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, models, min_loss)

if __name__=='__main__':
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    NewMain(args, opts)

    # feature_path = 'data/EgoMotion/features/02_01_walk/lab/1/feature_10frames.npy'
    # feature = np.load(feature_path)
    # print(feature.shape)
    # feature_path = 'data/EgoMotion/features/02_01_walk/lab/1/feature_01frames.npy'
    # feature = np.load(feature_path)
    # print(feature.shape)
    # data_root = "data/EgoMotion" 
    # config = "data/EgoMotion/meta_remy.yml"
    # image_tmpl = "{:04d}.jpg"
    # clip_len = 30
    # import torchvision
    # from PIL import Image
    # class Scale():
    #     def __init__(self, size, interpolation=Image.BILINEAR):
    #         self.worker = torchvision.transforms.Resize(size, interpolation)

    #     def __call__(self, img):
    #         return self.worker(img)
        
    # transforms = torchvision.transforms.Compose([Scale(256), torchvision.transforms.ToTensor()])
    # egodata = EgoMotionDataset(dataset_path=data_root,
    #                      config_path=config,
    #                      image_tmpl=image_tmpl,
    #                      transform=transforms,
    #                      clip_length=clip_len,
    #                      use_slam=True)
    # slam, pose_gt, clip = egodata[10]
    # print("slam shape: ", slam.shape)
    # print("pose_gt shape: ", pose_gt.shape)
    # print("clip shape: ", clip.shape)
    
    # feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
    # model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    # # if torch.cuda.is_available():
    # #     feature_extractor = nn.DataParallel(feature_extractor)
    # #     feature_extractor = feature_extractor.cuda()
    # #     model = nn.DataParallel(model)
    # #     model = model.cuda()
    # # prepare image for the model
    # inputs = feature_extractor(images=clip, return_tensors="pt")

    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     predicted_depth = outputs.predicted_depth
    # print("depth shape: ", predicted_depth.shape)