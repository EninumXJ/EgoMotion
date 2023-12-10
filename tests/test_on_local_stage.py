import sys
sys.path.append('..')
sys.path.append('.')
import os

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch3d.transforms as transforms 
import argparse

from model.timesformer.models.vit import TimeSformer
from model.proj import Projector
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation

from dataset.kinpoly_dataset import output2quat, output2matrix, KinPolyDataset, collate_function, NewKinPolyDataset
from finetune import set_random_seed
from MotionBERT.lib.utils.tools import *
from MotionBERT.lib.utils.learning import *
import prettytable
from tqdm import tqdm
from model.selformer import *
from model.loss import *
from train_on_kinpoly import depth_estimate
from utils.data_utils.process_kinpoly_qpos2smpl import qpos2smpl, qpos2smpl_vis, simpleqpos2smpl
from vis.vis_pose_matplot import show3Dpose_animation_smpl22, show3Dpose_animation, plot_single_pose

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/data/newhome/litianyi/logs/EgoMotion/", help="Path to the log file.")
    parser.add_argument("--config", type=str, default="config/DST_slam_train.yaml", help="Path to the config file.")
    parser.add_argument('-lr', '--local_resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-backbone', '--backbone', default='Timesformer', type=str, help='backbone of MotionBERT')
    parser.add_argument('-p', '--projector', choices=['linear', 'rnn'], type=str, help='projector type of EgoMotion')
    parser.add_argument('-v', '--vis_path', type=str, help='path of visualize results')
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    opts = parser.parse_args()
    return opts

def test_linear(args, localModel, localProj, val_loader, feature_extractor, backbone):
    print('INFO: Testing')
    num_joints = args.num_joints
    results_all_wo_gt = []; losses_all = {}; results_all_w_gt = []; gt_all = []
    root_all = []; predicted_result_local = []; gt_result_local = []; video_index = []; predicted_result_global = []
    losses_all['total'] = AverageMeter()
    losses_all['loss_quat_v'] = AverageMeter()
    losses_all['loss_quat'] = AverageMeter()
    localModel.eval()
    localProj.eval()
    with torch.no_grad():
        with tqdm(total=args.val_size) as pbar:
            for idx, (lrot_gt, img_clip, root, joint_rot, index) in enumerate(val_loader):
                N, C = img_clip.shape[:2]
                if torch.cuda.is_available():
                    lrot_gt = lrot_gt.cuda()[:,1:]  ## delete the first one(zero)
                    img_clip = img_clip.cuda()
                    root = root.cuda()
                    joint_rot = joint_rot.cuda()
                # -----------------------------------------------------------------------------------
                # Global Pose Predict
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
                # -----------------------------------------------------------------------------------
                # Local Pose Predict
                inputs = img_clip.permute(0, 2, 1, 3, 4)
                embeddings = localModel(inputs)  # (N, F, J, C) = (32, 10, 23, 3)
                
                if args.rot_represent == 'quat4d':
                    output = localProj(embeddings).reshape(N, -1, 23, 4)    # (N, T-1, 23, 4)
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
                    output = localProj(embeddings).reshape(N, -1, num_joints, 6)    # (N, T-1, num_joints, 6)
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
                # -----------------------------------------------------------------------------------
                # Fuse Two Stage Results to Get Final Results
                root = root.reshape(N, C, 7)
                root_trans = root[:, :, 0:3]   # N X 1 X 3
                root_rot_t0 = root[:, 0:1, 3:]     # N X 1 X 4
                # -----------------------------------------------------------------------------------
                # add ground truth
                root = root.reshape(N, C, 7)
                root_all.append(root.cpu())
                gt_result_local.append(joint_rot.cpu())
                # add results
                predicted_root_rot = predicted_root_rot.reshape(N, C, 4)
                root_predicted = torch.cat([root_trans, predicted_root_rot], dim=-1)
                video_index.extend(index)
                predicted_result_global.append(root_predicted.cpu())
                predicted_result_local.append(joint_euler_status.reshape(N,C,23,3).cpu())
                
                results_all_w_gt.append(predicted_3d_pos.cpu().numpy())
                gt_all.append(batch_gt.cpu().numpy())
                # -----------------------------------------------------------------------------------
                # update pbar
                if idx % 10 == 0:
                    pbar.update(10)
    
    gt_all = np.concatenate(gt_all)
    results_all_w_gt = np.concatenate(results_all_w_gt)
    print("test clips: ", results_all_w_gt.shape[0])
    num_test_clips = len(results_all_w_gt)
    e1_all_w_gt = np.zeros(num_test_clips)
    e2_all_w_gt = np.zeros(num_test_clips)
    oc = np.zeros(num_test_clips)

    final_results_1_wo_gt = []; final_results_2_wo_gt = []; final_results_1_w_gt = []; final_results_2_w_gt = []
    for idx in range(num_test_clips):
        gt = gt_all[idx]
        pred_w_gt = results_all_w_gt[idx]
        # Root-relative Errors
        # pred = pred - pred[:,0:1,:]
        # gt = gt - gt[:,0:1,:]
        # print("GT 3d pose: ", gt[0])
        # print("pred 3d pose: ", pred[0])
        err1_w_gt = mpjpe(pred_w_gt, gt)
        err2_w_gt = p_mpjpe(pred_w_gt, gt)
        
       
        e1_all_w_gt[idx] = np.mean(err1_w_gt)
        e2_all_w_gt[idx] = np.mean(err2_w_gt)
        oc[idx] += 1
       
        final_results_1_w_gt.append(e1_all_w_gt[idx])
        final_results_2_w_gt.append(e2_all_w_gt[idx])
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name']
    summary_table.add_row(['P1'])
    summary_table.add_row(['P2'])
    print(summary_table)
    
    e1 = np.mean(np.array(final_results_1_w_gt)) * 1000
    e2 = np.mean(np.array(final_results_2_w_gt)) * 1000
    print('Protocol #1 Error with GT (MPJPE):', e1, 'mm')
    print('Protocol #2 Error with GT (P-MPJPE):', e2, 'mm')
    print('----------')
    
    return torch.cat(root_all, dim=0), torch.cat(predicted_result_local, dim=0), \
           torch.cat(predicted_result_global, dim=0), torch.cat(gt_result_local, dim=0), video_index

def test_rnn(args, localModel, localProj, val_loader, feature_extractor, backbone):
    print('INFO: Testing')
    num_joints = args.num_joints
    results_all_wo_gt = []; losses_all = {}; results_all_w_gt = []; gt_all = []
    root_all = []; predicted_result_local = []; gt_result_local = []; video_index = []; predicted_result_global = []
    losses_all['total'] = AverageMeter()
    losses_all['loss_quat_v'] = AverageMeter()
    losses_all['loss_quat'] = AverageMeter()
    localModel.eval()
    localProj.eval()
    with torch.no_grad():
        with tqdm(total=args.val_size) as pbar:
            for idx, (lrot_gt, img_clip, root, joint_rot, index) in enumerate(val_loader):
                N, C = img_clip.shape[:2]
                if torch.cuda.is_available():
                    lrot_gt = lrot_gt.cuda()[:,1:]  ## delete the first one(zero)
                    img_clip = img_clip.cuda()
                    root = root.cuda()
                    joint_rot = joint_rot.cuda()
                # -----------------------------------------------------------------------------------
                # Global Pose Predict
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
                # -----------------------------------------------------------------------------------
                # Local Pose Predict
                inputs = img_clip.permute(0, 2, 1, 3, 4)
                embeddings = localModel(inputs)  # (N, F, J, C) = (32, 10, 23, 3)
                embeddings = embeddings.reshape(N, C-1, args.dim_feat).permute(1, 0, 2)  # (7, N, 144)
                # print("embeddings shape: ", embeddings.shape)
                initial_state = lrot_gt[:, 0:1, ...].reshape(N, 1, -1).permute(1, 0, 2)  # (N, 1, 24*6)
                h0 = torch.repeat_interleave(initial_state, args.rnn_hidden_layers, dim=0)  # (num_layers, N, 24*6)
                
                if args.rot_represent == 'quat4d':
                    output = localProj(embeddings, h0)[0].permute(1, 0, 2).reshape(N, -1, num_joints, 4)    # (N, T-1, 24, 4)
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
                    output = localProj(embeddings, h0)[0].permute(1, 0, 2).reshape(N, -1, num_joints, 6)    # (N, T-1, 24, 6)
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
                # -----------------------------------------------------------------------------------
                # Fuse Two Stage Results to Get Final Results
                root = root.reshape(N, C, 7)
                root_trans = root[:, :, 0:3]   # N X 1 X 3
                root_rot_t0 = root[:, 0:1, 3:]     # N X 1 X 4
                # -----------------------------------------------------------------------------------
                # add ground truth
                root = root.reshape(N, C, 7)
                root_all.append(root.cpu())
                gt_result_local.append(joint_rot.cpu())
                # add results
                predicted_root_rot = predicted_root_rot.reshape(N, C, 4)
                root_predicted = torch.cat([root_trans, predicted_root_rot], dim=-1)
                video_index.extend(index)
                predicted_result_global.append(root_predicted.cpu())
                predicted_result_local.append(joint_euler_status.reshape(N,C,-1,3).cpu())
                
                results_all_w_gt.append(predicted_3d_pos.cpu().numpy())
                gt_all.append(batch_gt.cpu().numpy())
                # -----------------------------------------------------------------------------------
                # update pbar
                if idx % 10 == 0:
                    pbar.update(10)
    
    gt_all = np.concatenate(gt_all)
    results_all_w_gt = np.concatenate(results_all_w_gt)
    print("test clips: ", results_all_w_gt.shape[0])
    num_test_clips = len(results_all_w_gt)
    e1_all_w_gt = np.zeros(num_test_clips)
    e2_all_w_gt = np.zeros(num_test_clips)
    oc = np.zeros(num_test_clips)

    final_results_1_wo_gt = []; final_results_2_wo_gt = []; final_results_1_w_gt = []; final_results_2_w_gt = []
    for idx in range(num_test_clips):
        gt = gt_all[idx]
        pred_w_gt = results_all_w_gt[idx]
        # Root-relative Errors
        # pred = pred - pred[:,0:1,:]
        # gt = gt - gt[:,0:1,:]
        # print("GT 3d pose: ", gt[0])
        # print("pred 3d pose: ", pred[0])
        err1_w_gt = mpjpe(pred_w_gt, gt)
        err2_w_gt = p_mpjpe(pred_w_gt, gt)
        
       
        e1_all_w_gt[idx] = np.mean(err1_w_gt)
        e2_all_w_gt[idx] = np.mean(err2_w_gt)
        oc[idx] += 1
       
        final_results_1_w_gt.append(e1_all_w_gt[idx])
        final_results_2_w_gt.append(e2_all_w_gt[idx])
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name']
    summary_table.add_row(['P1'])
    summary_table.add_row(['P2'])
    print(summary_table)
    
    e1 = np.mean(np.array(final_results_1_w_gt)) * 1000
    e2 = np.mean(np.array(final_results_2_w_gt)) * 1000
    print('Protocol #1 Error with GT (MPJPE):', e1, 'mm')
    print('Protocol #2 Error with GT (P-MPJPE):', e2, 'mm')
    print('----------')
    
    return torch.cat(root_all, dim=0), torch.cat(predicted_result_local, dim=0), \
           torch.cat(predicted_result_global, dim=0), torch.cat(gt_result_local, dim=0), \
           gt_all, results_all_w_gt, video_index

def test_transformer(args, backbone, transformer, val_loader):
    print('INFO: Testing')
    num_joints = args.num_joints
    results_all_wo_gt = []; losses_all = {}; results_all_w_gt = []; gt_all = []
    root_all = []; predicted_result_local = []; gt_result_local = []; video_index = []; predicted_result_global = []
    losses_all['total'] = AverageMeter()
    losses_all['loss_quat_v'] = AverageMeter()
    losses_all['loss_quat'] = AverageMeter()
    transformer.eval()
    num_joints = args.num_joints           
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for idx, (lrot_gt, img_clip, root, joint_rot, index) in enumerate(val_loader):
                N, L, C = img_clip.shape[0:3]
                # print("input shape: ", batch_input.shape)
                if torch.cuda.is_available():
                    img_clip = img_clip.cuda() # B,15,8,3,224,224
                    lrot_gt = lrot_gt.cuda()[:,1:,...]  #B,15,13,6
                    joint_rot = joint_rot.cuda() # B,16,12,3
                    root = root.cuda()  # B,16,7
                if isinstance(transformer, torch.nn.DataParallel):
                    transformer = transformer.module
                else:
                    transformer = transformer.model
                
                inputs = img_clip.reshape(-1, C, 3, 224, 224).permute(0, 2, 1, 3, 4)
                feature = backbone(inputs).reshape(N, L, -1)  # (N, L, 256)
                src_mask = None
                joint_rot_6d = transforms.matrix_to_rotation_6d(transforms.euler_angles_to_matrix(joint_rot, convention='ZYX'))[:, 0:1, ...]  # (N, 1, 12, 6)
                root_rot_6d = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(root[:,0:1,3:]))[:,:,None,:]  # (N, 1, 1, 6)
                joint_rot_6d = torch.cat([root_rot_6d, joint_rot_6d], dim=2)   # (N, 1, 13, 6)
                
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
                    print("output: ", output)
                    predicted_joint_rot6d = output    # (N, T, num_joints, 6)
                    predicted_joint_mat = transforms.rotation_6d_to_matrix(predicted_joint_rot6d)  # (N, T, num_joints, 3, 3)
                    root_quat_initial = root[:,0:1,None,3:]  # (N, 1, 1, 4)
                    predicted_root_rot_diff = transforms.matrix_to_quaternion(predicted_joint_mat[:,:,0:1,...])  # (N, T, 1, 4)
                    predicted_others_rot_diff = predicted_joint_mat[:,:,1:,...]  # (N, T, num_joints-1, 3, 3)
                    predicted_joint_matrix = output2matrix(predicted_others_rot_diff, joint_rot, mode='to-initial')  # (N, T, 12, 3, 3)
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
                    batch_gt = batch_gt.reshape(N, -1, 22, 3)[:,1:,...]   # (N, T, 22, 3)
                    predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)[:,1:,...]   # (N, T, 22, 3)
                
                # -----------------------------------------------------------------------------------
                # Fuse Two Stage Results to Get Final Results
                root = root.reshape(N, L+1, 7)
                root_trans = root[:, :, 0:3]   # N X L+1 X 3
                root_rot_t0 = root[:, 0:1, 3:]     # N X 1 X 4
                # -----------------------------------------------------------------------------------
                # add ground truth
                root = root.reshape(N, L+1, 7)
                root_all.append(root.cpu())
                gt_result_local.append(joint_rot.cpu())
                # add results
                predicted_root_rot = predicted_root_rot.reshape(N, L+1, 4)
                root_predicted = torch.cat([root_trans, predicted_root_rot], dim=-1)
                video_index.extend(index)
                predicted_result_global.append(root_predicted.cpu())
                predicted_result_local.append(joint_euler_status.reshape(N,L+1,-1,3).cpu())
                
                results_all_w_gt.append(predicted_3d_pos.cpu().numpy())
                gt_all.append(batch_gt.cpu().numpy())
                # -----------------------------------------------------------------------------------
                # update pbar
                if idx % 10 == 0:
                    pbar.update(10)
    
    gt_all = np.concatenate(gt_all)
    results_all_w_gt = np.concatenate(results_all_w_gt)
    print("test clips: ", results_all_w_gt.shape[0])
    num_test_clips = len(results_all_w_gt)
    e1_all_w_gt = np.zeros(num_test_clips)
    e2_all_w_gt = np.zeros(num_test_clips)
    oc = np.zeros(num_test_clips)

    final_results_1_wo_gt = []; final_results_2_wo_gt = []; final_results_1_w_gt = []; final_results_2_w_gt = []
    for idx in range(num_test_clips):
        gt = gt_all[idx]
        pred_w_gt = results_all_w_gt[idx]
        # Root-relative Errors
        # pred = pred - pred[:,0:1,:]
        # gt = gt - gt[:,0:1,:]
        # print("GT 3d pose: ", gt[0])
        # print("pred 3d pose: ", pred[0])
        err1_w_gt = mpjpe(pred_w_gt, gt)
        err2_w_gt = p_mpjpe(pred_w_gt, gt)
        
       
        e1_all_w_gt[idx] = np.mean(err1_w_gt)
        e2_all_w_gt[idx] = np.mean(err2_w_gt)
        oc[idx] += 1
       
        final_results_1_w_gt.append(e1_all_w_gt[idx])
        final_results_2_w_gt.append(e2_all_w_gt[idx])
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name']
    summary_table.add_row(['P1'])
    summary_table.add_row(['P2'])
    print(summary_table)
    
    e1 = np.mean(np.array(final_results_1_w_gt)) * 1000
    e2 = np.mean(np.array(final_results_2_w_gt)) * 1000
    print('Protocol #1 Error with GT (MPJPE):', e1, 'mm')
    print('Protocol #2 Error with GT (P-MPJPE):', e2, 'mm')
    print('----------')
    
    return torch.cat(root_all, dim=0), torch.cat(predicted_result_local, dim=0), \
           torch.cat(predicted_result_global, dim=0), torch.cat(gt_result_local, dim=0), \
           gt_all, results_all_w_gt, video_index

def main(opts, args):
    img_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224)),
                    torchvision.transforms.ToTensor()])
    test_data = KinPolyDataset(dataset_path=args.data_root,
                        config_path=args.config,
                        image_tmpl=args.image_tmpl,
                        transform=img_transforms,
                        clip_length=args.clip_len,
                        mode='test',
                        use_slam=False,
                        rot=args.rot_represent,
                        coordinate='local',
                        if_sample=args.if_sample,
                        num_of_keypoints=args.num_joints)
    test_dataset, _ = random_split(test_data, [args.val_size, len(test_data) - args.val_size])
    val_dataset = test_dataset
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=False, collate_fn=collate_function)
    # --------------------------------------------------------------------------
    # Model
    num_frames = int(args.clip_len / 2) if args.if_sample else args.clip_len
    # globalModel = TimeSformer(img_size=224, num_classes=224, num_frames=8, attention_type='divided_space_time',
    #                           pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    localModel = TimeSformer(img_size=224, num_classes=args.dim_feat*(num_frames-1), num_frames=args.clip_len, attention_type='divided_space_time',
                             pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    # ----- Local net ----------
    num_joints = args.num_joints
    # projector
    if args.rot_represent == 'rot6d':
        if opts.projector == 'linear':
            proj = Projector(args.dim_feat*7, num_joints*6*7, 512, 1024)
        elif opts.projector == 'rnn':
            proj = nn.RNN(args.dim_feat, num_joints*6, args.rnn_hidden_layers) 
    elif args.rot_represent == 'quat4d':
        if opts.projector == 'linear':
            proj = Projector(args.dim_feat*7, num_joints*4*7, 512)
        elif opts.projector == 'rnn':
            proj = nn.RNN(args.dim_feat, num_joints*4, args.rnn_hidden_layers)
    model_params = 0
    for parameter in localModel.parameters():
        model_params = model_params + parameter.numel()
    for parameter in proj.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    
    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    # init depeth estimator
    if args.use_depth:
        feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
        backbone = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
        if torch.cuda.is_available():
            backbone = nn.DataParallel(backbone)
            backbone = backbone.cuda()
    else:
        feature_extractor = None
        backbone = None
    # --------------------------------------------------------------------------
    #
    if torch.cuda.is_available():
        localModel = nn.DataParallel(localModel)
        localModel = localModel.cuda()
        proj = nn.DataParallel(proj)
        proj = proj.cuda()
    # --------------------------------------------------------------------------
    # resume from checkpoints
    chk_filename = opts.local_resume
    print("Loading local resume", chk_filename)
    # ----- Local --------
    checkpoint2 = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    localModel.load_state_dict(checkpoint2['backbone'], strict=True)
    proj.load_state_dict(checkpoint2['proj'], strict=True)
    
    if opts.projector == 'linear':
        root_all, predicted_result_local, predicted_result_global, gt_result_local, video_index = \
            test_linear(args, localModel, proj, val_loader, feature_extractor, backbone)
    elif opts.projector == 'rnn':
        root_all, predicted_result_local, predicted_result_global, gt_result_local, gt_all, results_all_w_gt, video_index = \
            test_rnn(args, localModel, proj, val_loader, feature_extractor, backbone)

    exp_dir = os.path.join(opts.log_dir, args.experiment_name, 'test')
    vis_res_folder = os.path.join(exp_dir, opts.vis_path)
    if not os.path.exists(vis_res_folder):
        os.makedirs(vis_res_folder)
    # print("predicted_result_quat_global: ", predicted_result_global[0,:,3:])
    # print("predicted_result_trans_global: ", predicted_result_global[0,:,0:3])
    # print("GT_result_quat_global: ", root_all[0,:,3:])
    # print("GT_result_trans_global: ", root_all[0,:,0:3])
    pose_to_plot = np.concatenate([gt_all[0:1], results_all_w_gt[0:1]], axis=0)
    # np.save("result_3d_pos.npy", pose_to_plot)
    # pose_to_plot = np.load("result_3d_pos.npy")
    # print("pose_to_plot shape: ", pose_to_plot.shape)
    video_name = video_index[0][0]
    # video_name = '1219_take_40'
    frame_index = video_index[0][1]
    # frame_index = 232
    dir_name = video_name + '_' + str(frame_index)
    # dest_vis_path = os.path.join(vis_res_folder, dir_name)
    # if not os.path.exists(dest_vis_path):
    #     os.makedirs(dest_vis_path)
    # for frame in range(pose_to_plot.shape[1]):
    #     plot_single_pose(pose_to_plot[0][frame], frame, dest_vis_path, prefix='predict')
    # qpos2smpl_vis(predicted_result_local[0,...], predicted_result_global[0,:,3:], predicted_result_global[0,:,0:3], vis_res_folder, 'predicted')
    # qpos2smpl_vis(gt_result_local[0,...], root_all[0,:,3:], root_all[0,:,0:3], vis_res_folder, 'ground_truth')
    for i in range(root_all.shape[0]):
        video_name = video_index[i][0]
        frame_index = video_index[i][1]
        dir_name = video_name + '_' + str(frame_index)
        vis_res_folder_clip = os.path.join(vis_res_folder, dir_name)
        pose_to_plot = np.concatenate([gt_all[i:i+1], results_all_w_gt[i:i+1]], axis=0)
        if not os.path.exists(vis_res_folder_clip):
            os.makedirs(vis_res_folder_clip)
        for frame in range(pose_to_plot.shape[1]):
            plot_single_pose(pose_to_plot[0][frame], frame, vis_res_folder_clip, prefix='gt')
            plot_single_pose(pose_to_plot[1][frame], frame, vis_res_folder_clip, prefix='predict')
        # qpos2smpl_vis(predicted_result_local[i,...], predicted_result_global[i,:,3:], predicted_result_global[i,:,0:3], vis_res_folder_clip, 'predicted')
        # qpos2smpl_vis(gt_result_local[i,...], root_all[i,:,3:], root_all[i,:,0:3], vis_res_folder_clip, 'ground_truth')

def NewMain(opts, args):
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
    test_dataset, _ = random_split(test_data, [args.val_size, len(test_data) - args.val_size])
    val_dataset = test_dataset
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=4, pin_memory=False, collate_fn=collate_function)
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
    if torch.cuda.is_available():
        transformer = nn.DataParallel(transformer)
        transformer = transformer.cuda()
        backbone = nn.DataParallel(backbone)
        backbone = backbone.cuda()

    checkpoint = None
    # ---------------------------------------------------------------------------
    # TODO resume from checkpoint
    if opts.local_resume:
        chk_filename = opts.local_resume
        print("Loading resume", chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        transformer.load_state_dict(checkpoint['transformer'], strict=True)
    root_all, predicted_result_local, predicted_result_global, gt_result_local, gt_all, results_all_w_gt, video_index = \
            test_transformer(args, backbone, transformer, val_loader)  
    exp_dir = os.path.join(opts.log_dir, args.experiment_name, 'test')
    vis_res_folder = os.path.join(exp_dir, opts.vis_path)
    if not os.path.exists(vis_res_folder):
        os.makedirs(vis_res_folder)
    # print("predicted_result_quat_global: ", predicted_result_global[0,:,3:])
    # print("predicted_result_trans_global: ", predicted_result_global[0,:,0:3])
    # print("GT_result_quat_global: ", root_all[0,:,3:])
    # print("GT_result_trans_global: ", root_all[0,:,0:3])
    pose_to_plot = np.concatenate([gt_all[0:1], results_all_w_gt[0:1]], axis=0)
    # np.save("result_3d_pos.npy", pose_to_plot)
    # pose_to_plot = np.load("result_3d_pos.npy")
    # print("pose_to_plot shape: ", pose_to_plot.shape)
    video_name = video_index[0][0]
    # video_name = '1219_take_40'
    frame_index = video_index[0][1]
    # frame_index = 232
    print(video_index)
    dir_name = video_name + '_' + str(frame_index)
    # dest_vis_path = os.path.join(vis_res_folder, dir_name)
    # if not os.path.exists(dest_vis_path):
    #     os.makedirs(dest_vis_path)
    # for frame in range(pose_to_plot.shape[1]):
    #     plot_single_pose(pose_to_plot[0][frame], frame, dest_vis_path, prefix='predict')
    # qpos2smpl_vis(predicted_result_local[0,...], predicted_result_global[0,:,3:], predicted_result_global[0,:,0:3], vis_res_folder, 'predicted')
    # qpos2smpl_vis(gt_result_local[0,...], root_all[0,:,3:], root_all[0,:,0:3], vis_res_folder, 'ground_truth')
    for i in range(root_all.shape[0]):
        video_name = video_index[i][0]
        frame_index = video_index[i][1]
        print("video_name: ", video_name)
        print("frame_index: ", frame_index)
        dir_name = str(video_name) + '_' + str(frame_index)
        vis_res_folder_clip = os.path.join(vis_res_folder, dir_name)
        pose_to_plot = np.concatenate([gt_all[i:i+1], results_all_w_gt[i:i+1]], axis=0)
        if not os.path.exists(vis_res_folder_clip):
            os.makedirs(vis_res_folder_clip)
        for frame in range(pose_to_plot.shape[1]):
            plot_single_pose(pose_to_plot[0][frame], frame, vis_res_folder_clip, prefix='gt')
            plot_single_pose(pose_to_plot[1][frame], frame, vis_res_folder_clip, prefix='predict')

if __name__ == '__main__':
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    NewMain(opts, args)