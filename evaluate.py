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
    parser.add_argument("--config", type=str, default="config/DST_slam_train.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-backbone', '--backbone', default='Timesformer', type=str, help='backbone of MotionBERT')
    parser.add_argument('-v', '--vis_path', type=str, help='path of visualize results')
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    opts = parser.parse_args()
    return opts

def evaluate(args, model, proj, val_loader, feature_extractor=None, backbone=None):
    print('INFO: Testing')
    results_all = []; gt_all = []; losses_all = {}
    root_all = []; predicted_result = []; gt_result = []; video_index = []
    losses_all['total'] = AverageMeter()
    losses_all['loss_quat_v'] = AverageMeter()
    losses_all['loss_quat'] = AverageMeter()
    model.eval()
    proj.eval()           
    with torch.no_grad():
        with tqdm(total=args.val_size) as pbar:
            for idx, (lrot_gt, img_clip, root, joint_rot, index) in enumerate(val_loader):
                N, C = img_clip.shape[:2]
                # print("index: ", index)
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
                    output = proj(embeddings).reshape(N, -1, 23, 6)    # (N, T-1, 23, 6)
                    predicted_joint_rot6f_diff = output
                    predicted_joint_matrix = output2matrix(predicted_joint_rot6f_diff, joint_rot)  # (N, T, 23, 4)
                    # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                    # output quat to euler
                    joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX').reshape(-1, 23, 3)
                    root = root.reshape(-1, 7)
                    predicted_3d_pos = qpos2smpl(joint_euler_status, root[:, 3:], root[:, 0:3])
                    # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                    batch_gt = qpos2smpl(joint_rot.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                    batch_gt = batch_gt.reshape(N, -1, 22, 3)
                    predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)

                loss_quat_ = loss_quat(output, lrot_gt)
                loss_quat_v_ = loss_quat_v(output, lrot_gt)
                loss_total = loss_quat_ * 5 + \
                                args.lambda_scale  * loss_quat_v_

                losses_all['loss_quat'].update(loss_quat_.item(), N)
                losses_all['loss_quat_v'].update(loss_quat_v_.item(), N)
                losses_all['total'].update(loss_total.item(), N)
                root = root.reshape(N, C, 7)
                root_all.append(root.cpu())
                predicted_result.append(joint_euler_status.reshape(N,C,23,3).cpu())
                gt_result.append(joint_rot.cpu())
                video_index.extend(index)
                results_all.append(predicted_3d_pos.cpu().numpy())
                gt_all.append(batch_gt.cpu().numpy())
                if idx % 10 == 0:
                    pbar.update(10)
    
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
    return torch.cat(root_all, dim=0), torch.cat(predicted_result, dim=0), \
           torch.cat(gt_result, dim=0), video_index



def test(opts, args):
    img_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224)),
                    torchvision.transforms.ToTensor()])
    test_data = KinPolyDataset(dataset_path=args.data_root,
                        config_path=args.config,
                        image_tmpl=args.image_tmpl,
                        transform=img_transforms,
                        clip_length=args.clip_len,
                        mode='test',
                        use_slam=True,
                        rot=args.rot_represent)
    test_dataset, _ = random_split(test_data, [args.val_size, len(test_data) - args.val_size])
    val_dataset = test_dataset
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=False, collate_fn=collate_function)
    # --------------------------------------------------------------------------
    # Model
    model = TimeSformer(img_size=224, num_classes=224, num_frames=8, attention_type='divided_space_time',
                        pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    
    # projector
    if args.rot_represent == 'rot6d':
        proj = Projector(224, 23*6*7, 512, 1024)
    elif args.rot_represent == 'quat4d':
        proj = Projector(224, 23*4*7, 512)
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
    # resume from checkpoint
    if opts.resume:
        chk_filename = opts.resume
        print("Loading resume", chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['backbone'], strict=True)
        proj.load_state_dict(checkpoint['proj'], strict=True)

    root_all, predicted_result, gt_result, video_index = evaluate(args, model, proj, val_loader, feature_extractor, backbone)
    print("root shape: ", root_all.shape)
    print("predicted_result shape: ", predicted_result.shape)
    print("gt_result shape: ", gt_result.shape)
    print(video_index)
    print("video_name: ", video_index[0][0])
    print("frame index: ", video_index[0][1])
    # for i in range(root_all.shape[0]):
    #     video_name = video_index[i][0]
    #     frame_index = video_index[i][1]
    #     dir_name = video_name + '_' + str(frame_index)
    #     vis_res_folder = os.path.join(opts.vis_path, dir_name)
    #     qpos2smpl_vis(predicted_result[i,...], root_all[i,:,3:], root_all[i,:,0:3], vis_res_folder, 'predicted')
    #     qpos2smpl_vis(gt_result[i,...], root_all[i,:,3:], root_all[i,:,0:3], vis_res_folder, 'ground_truth')
    vis_res_folder = opts.vis_path
    root_zero_quat = torch.zeros_like(root_all[0,:,3:])
    # root_zero_quat[:,0] = 0.5
    # root_zero_quat[:,1] = 0.5
    # root_zero_quat[:,2] = 0.5
    # root_zero_quat[:,3] = 0.5
    ### 这组四元数能够确保人物朝向和之前的一致
    root_zero_quat[:,2] = 0.707
    root_zero_quat[:,3] = 0.707
    root_zero_trans = torch.zeros_like(root_all[0,:,0:3])
    root_zero_trans[:,2] = 0.777
    # qpos2smpl_vis(predicted_result[0,...], root_all[0,:,3:], root_all[0,:,0:3], vis_res_folder, 'predicted')
    # qpos2smpl_vis(gt_result[0,...], root_all[0,:,3:], root_all[0,:,0:3], vis_res_folder, 'ground_truth')
    ## normalize: 为了说明全局姿态和局部姿态的区别
    # qpos2smpl_vis(gt_result[0,...], root_zero_quat, root_zero_trans, vis_res_folder, 'zero_trans_zero_rotation')
    # qpos2smpl_vis(gt_result[0,...], root_all[0,:,3:], root_zero_trans, vis_res_folder, 'zero_trans')

if __name__=='__main__':
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    test(opts, args)
    