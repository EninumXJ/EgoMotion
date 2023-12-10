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

from dataset.kinpoly_dataset import output2quat, output2matrix, KinPolyDataset, collate_function
from finetune import set_random_seed
from MotionBERT.lib.utils.tools import *
from MotionBERT.lib.utils.learning import *
import prettytable
from tqdm import tqdm
from model.loss import *
from train_on_kinpoly import depth_estimate
from utils.data_utils.process_kinpoly_qpos2smpl import qpos2smpl, qpos2smpl_vis

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/data/newhome/litianyi/logs/EgoMotion/", help="Path to the log file.")
    parser.add_argument("--config", type=str, default="config/DST_slam_train.yaml", help="Path to the config file.")
    parser.add_argument('-lr', '--local_resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-gr', '--global_resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-backbone', '--backbone', default='Timesformer', type=str, help='backbone of MotionBERT')
    parser.add_argument('-v', '--vis_path', type=str, help='path of visualize results')
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    opts = parser.parse_args()
    return opts

def test(args, globalModel, localModel, localProj, transProj, rotProj, val_loader, feature_extractor, backbone):
    print('INFO: Testing')
    results_all_wo_gt = []; losses_all = {}; results_all_w_gt = []; gt_all = []
    root_all = []; predicted_result_local = []; gt_result_local = []; video_index = []; predicted_result_global = []
    losses_all['total'] = AverageMeter()
    losses_all['loss_quat_v'] = AverageMeter()
    losses_all['loss_quat'] = AverageMeter()
    globalModel.eval(); localModel.eval();
    localProj.eval(); transProj.eval(); rotProj.eval()
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
                embeddings = globalModel(inputs)  # (N, F, J, C) = (32, 10, 23, 3)
                predicted_trans_diff = transProj(embeddings).reshape(N, -1, 3)    # (N, T-1, 3)
                if args.rot_represent == 'quat4d':
                    predicted_rot_diff = rotProj(embeddings).reshape(N, -1, 4)    # (N, T-1, 4)
                elif args.rot_represent == 'rot6d':
                    predicted_rot_diff = rotProj(embeddings).reshape(N, -1, 6)   # (N, T-1, 6)
                    predicted_rot_diff = transforms.matrix_to_quaternion(transforms.rotation_6d_to_matrix(predicted_rot_diff))   # (N, T-1, 4)
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
                   
                elif args.rot_represent == 'rot6d':
                    output = localProj(embeddings).reshape(N, -1, 23, 6)    # (N, T-1, 23, 6)
                    predicted_joint_rot6f_diff = output
                    predicted_joint_matrix = output2matrix(predicted_joint_rot6f_diff, joint_rot)  # (N, T, 23, 4)
                    # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                    # output quat to euler
                    joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX').reshape(-1, 23, 3)
                # -----------------------------------------------------------------------------------
                # Fuse Two Stage Results to Get Final Results
                # root = root.reshape(-1, 1, 7)
                root_trans_t0 = root[:, 0:1, 0:3]   # N X 1 X 3
                root_trans = torch.cumsum(torch.cat([root_trans_t0, predicted_trans_diff], dim=1), dim=1)  # N X T X 3  在时序上进行累加
                root_rot_t0 = root[:, 0:1, 3:]     # N X 1 X 4
                root_rot = torch.cat([root_rot_t0, predicted_rot_diff], dim=1)  # N X T X 4
                for t in range(1, root_rot.shape[1]):
                    root_rot[:,t,...] = transforms.quaternion_multiply(root_rot[:,t-1,...], root_rot[:,t,...])
                root_trans = root_trans.reshape(-1, 3)
                root_rot = root_rot.reshape(-1, 4)
                root = root.reshape(-1, 7)
                print("root_trans predicted: ", root_trans)
                print("root_trans gt: ", root[:, 0:3])
                print("root_rot predicted: ", root_rot)
                print("root_rot gt: ", root[:, 3:])
                predicted_3d_pos_wo_gt = qpos2smpl(joint_euler_status, root_rot, root_trans)   ## 3d pose wo GT root pose
                predicted_3d_pos_w_gt = qpos2smpl(joint_euler_status, root[:, 3:], root[:, 0:3])   ## 3d pose with GT root pose
                # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                batch_gt = qpos2smpl(joint_rot.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                batch_gt = batch_gt.reshape(N, -1, 22, 3)
                predicted_3d_pos_wo_gt = predicted_3d_pos_wo_gt.reshape(N, -1, 22, 3)
                predicted_3d_pos_w_gt = predicted_3d_pos_w_gt.reshape(N, -1, 22, 3)
                print("predict 3d pos: ", predicted_3d_pos_wo_gt[0])
                print("GT 3d pos: ", batch_gt[0])
                # -----------------------------------------------------------------------------------
                # add ground truth
                root = root.reshape(N, C, 7)
                root_all.append(root.cpu())
                gt_result_local.append(joint_rot.cpu())
                # add results
                root_trans = root_trans.reshape(N, C, 3)
                root_rot = root_rot.reshape(N, C, 4)
                root_predicted = torch.cat([root_trans, root_rot], dim=-1)
                video_index.extend(index)
                predicted_result_global.append(root_predicted.cpu())
                predicted_result_local.append(joint_euler_status.reshape(N,C,23,3).cpu())
                results_all_wo_gt.append(predicted_3d_pos_wo_gt.cpu().numpy())
                results_all_w_gt.append(predicted_3d_pos_w_gt.cpu().numpy())
                gt_all.append(batch_gt.cpu().numpy())
                # -----------------------------------------------------------------------------------
                # update pbar
                if idx % 10 == 0:
                    pbar.update(10)
    
    gt_all = np.concatenate(gt_all)
    results_all_wo_gt = np.concatenate(results_all_wo_gt)
    results_all_w_gt = np.concatenate(results_all_w_gt)
    print("test clips: ", results_all_wo_gt.shape[0])
    num_test_clips = len(results_all_wo_gt)
    e1_all_wo_gt = np.zeros(num_test_clips)
    e2_all_wo_gt = np.zeros(num_test_clips)
    e1_all_w_gt = np.zeros(num_test_clips)
    e2_all_w_gt = np.zeros(num_test_clips)
    oc = np.zeros(num_test_clips)

    final_results_1_wo_gt = []; final_results_2_wo_gt = []; final_results_1_w_gt = []; final_results_2_w_gt = []
    for idx in range(num_test_clips):
        gt = gt_all[idx]
        pred_wo_gt = results_all_wo_gt[idx]
        pred_w_gt = results_all_w_gt[idx]
        # Root-relative Errors
        # pred = pred - pred[:,0:1,:]
        # gt = gt - gt[:,0:1,:]
        # print("GT 3d pose: ", gt[0])
        # print("pred 3d pose: ", pred[0])
        err1_wo_gt = mpjpe(pred_wo_gt, gt)
        err2_wo_gt = p_mpjpe(pred_wo_gt, gt)
        err1_w_gt = mpjpe(pred_w_gt, gt)
        err2_w_gt = p_mpjpe(pred_w_gt, gt)
        
        e1_all_wo_gt[idx] = np.mean(err1_wo_gt)
        e2_all_wo_gt[idx] = np.mean(err2_wo_gt)
        e1_all_w_gt[idx] = np.mean(err1_w_gt)
        e2_all_w_gt[idx] = np.mean(err2_w_gt)
        oc[idx] += 1
        final_results_1_wo_gt.append(e1_all_wo_gt[idx])
        final_results_2_wo_gt.append(e2_all_wo_gt[idx])
        final_results_1_w_gt.append(e1_all_w_gt[idx])
        final_results_2_w_gt.append(e2_all_w_gt[idx])
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name']
    
    summary_table.add_row(['P1'])
    summary_table.add_row(['P2'])
    print(summary_table)
    e1 = np.mean(np.array(final_results_1_wo_gt)) * 1000
    e2 = np.mean(np.array(final_results_2_wo_gt)) * 1000
    print('Protocol #1 Error without GT (MPJPE):', e1, 'mm')
    print('Protocol #2 Error without GT (P-MPJPE):', e2, 'mm')
    print('----------')
    e1 = np.mean(np.array(final_results_1_w_gt)) * 1000
    e2 = np.mean(np.array(final_results_2_w_gt)) * 1000
    print('Protocol #1 Error with GT (MPJPE):', e1, 'mm')
    print('Protocol #2 Error with GT (P-MPJPE):', e2, 'mm')
    print('----------')
    
    return torch.cat(root_all, dim=0), torch.cat(predicted_result_local, dim=0), \
           torch.cat(predicted_result_global, dim=0), torch.cat(gt_result_local, dim=0), video_index
                
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
                        coordinate='local')
    test_dataset, _ = random_split(test_data, [args.val_size, len(test_data) - args.val_size])
    val_dataset = test_dataset
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=False, collate_fn=collate_function)
    # --------------------------------------------------------------------------
    # Model
    globalModel = TimeSformer(img_size=224, num_classes=224, num_frames=8, attention_type='divided_space_time',
                              pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    localModel = TimeSformer(img_size=224, num_classes=224, num_frames=8, attention_type='divided_space_time',
                             pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    # ----- Local net ----------
    # projector
    if args.rot_represent == 'rot6d':
        localProj = Projector(224, 23*6*7, 512, 1024)
    elif args.rot_represent == 'quat4d':
        localProj = Projector(224, 23*4*7, 512)
    # ----- Global  net --------
    transProj = Projector(224, 3*7, 128, 32)
    if args.rot_represent == 'rot6d':
        rotProj = Projector(224, 6*7, 128, 64)
    elif args.rot_represent == 'quat4d':
        rotProj = Projector(224, 4*7, 128, 64)
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
        globalModel = nn.DataParallel(globalModel)
        globalModel = globalModel.cuda()
        transProj = nn.DataParallel(transProj)
        transProj = transProj.cuda()
        rotProj = nn.DataParallel(rotProj)
        rotProj = rotProj.cuda()
        localModel = nn.DataParallel(localModel)
        localModel = localModel.cuda()
        localProj = nn.DataParallel(localProj)
        localProj = localProj.cuda()
    # --------------------------------------------------------------------------
    # resume from checkpoints
    chk_filename_global = opts.global_resume
    chk_filename_local = opts.local_resume
    print("Loading global resume", chk_filename_global)
    print("Loading local resume", chk_filename_local)
    # ----- Global --------
    checkpoint1 = torch.load(chk_filename_global, map_location=lambda storage, loc: storage)
    globalModel.load_state_dict(checkpoint1['backbone'], strict=True)
    transProj.load_state_dict(checkpoint1['proj_trans'], strict=True)
    rotProj.load_state_dict(checkpoint1['proj_rot'], strict=True)
    # ----- Local --------
    checkpoint2 = torch.load(chk_filename_local, map_location=lambda storage, loc: storage)
    localModel.load_state_dict(checkpoint2['backbone'], strict=True)
    localProj.load_state_dict(checkpoint2['proj'], strict=True)
    
    root_all, predicted_result_local, predicted_result_global, gt_result_local, video_index = \
        test(args, globalModel, localModel, localProj, transProj, 
             rotProj, val_loader, feature_extractor, backbone)

    exp_dir = os.path.join(opts.log_dir, args.experiment_name)
    vis_res_folder = os.path.join(exp_dir, opts.vis_path)
    if not os.path.exists(vis_res_folder):
        os.makedirs(vis_res_folder)
    print("predicted_result_quat_global: ", predicted_result_global[0,:,3:])
    print("predicted_result_trans_global: ", predicted_result_global[0,:,0:3])
    print("GT_result_quat_global: ", root_all[0,:,3:])
    print("GT_result_trans_global: ", root_all[0,:,0:3])
    # qpos2smpl_vis(predicted_result_local[0,...], predicted_result_global[0,:,3:], predicted_result_global[0,:,0:3], vis_res_folder, 'predicted')
    # qpos2smpl_vis(gt_result_local[0,...], root_all[0,:,3:], root_all[0,:,0:3], vis_res_folder, 'ground_truth')


if __name__ == '__main__':
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    main(opts, args)