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
from dataset.kinpoly_dataset import output2quat, output2matrix, KinPolyDataset
from model.proj import Projector
from utils.data_utils.process_kinpoly_qpos2smpl import qpos2smpl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/data/newhome/litianyi/logs/EgoMotion/", help="Path to the log file.")
    parser.add_argument("--config", type=str, default="config/DST_slam_train.yaml", help="Path to the config file.")
    # parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-backbone', '--backbone', default='Timesformer', type=str, help='backbone of MotionBERT')
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    opts = parser.parse_args()
    return opts

def evaluate(args, model, proj, val_loader, feature_extractor=None, backbone=None):
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
                    output = proj(embeddings).reshape(N, -1, 23, 6)    # (N, T-1, 23, 6)
                    predicted_joint_rot6f_diff = output
                    with torch.no_grad():
                        predicted_joint_matrix = output2matrix(predicted_joint_rot6f_diff, joint_rot)  # (N, T, 23, 4)
                        # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                        # output quat to euler
                        joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX')
                        root = root.reshape(-1, 7)
                        predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                        # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                        batch_gt = qpos2smpl(joint_rot.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
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

def train_epoch(args, model, proj, train_loader, losses, optimizer, feature_extractor=None, backbone=None):
    model.train()
    proj.train()
    with tqdm(total=len(train_loader)) as pbar:
        for idx, (lrot_gt, img_clip, root, joint_rot) in enumerate(train_loader):
            N, C = img_clip.shape[0:2]
            # print("input shape: ", batch_input.shape)
            if torch.cuda.is_available():
                img_clip = img_clip.cuda()
                lrot_gt = lrot_gt.cuda()[:, 1:]
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
            embeddings = model(inputs)
            # (N, F, J, C) = (32, 10, 17, 3)
            if args.rot_represent == 'quat4d':
                predicted_joint_rot = proj(embeddings).reshape(N, -1, 23, 4)    # (N, T-1, 17, 4)
                predicted_joint_quat = output2quat(predicted_joint_rot, joint_rot)  # (N, T, 23, 4)
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
                predicted_joint_rot = proj(embeddings).reshape(N, -1, 23, 6)    # (N, T-1, 17, 6)
                predicted_joint_matrix = output2matrix(predicted_joint_rot, joint_rot)  # (N, T, 23, 4)
                # print("predicted_joint_quat shape: ", predicted_joint_quat.shape)
                # output quat to euler
                joint_euler_status = transforms.matrix_to_euler_angles(predicted_joint_matrix, convention='ZYX')
                root = root.reshape(-1, 7)
                predicted_3d_pos = qpos2smpl(joint_euler_status.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                # print("predicted_3d_pos shape: ", predicted_3d_pos.shape)
                batch_gt = qpos2smpl(joint_rot.reshape(-1, 23, 3), root[:, 3:], root[:, 0:3])
                batch_gt = batch_gt.reshape(N, -1, 22, 3)
                predicted_3d_pos = predicted_3d_pos.reshape(N, -1, 22, 3)
           
            optimizer.zero_grad()
            loss_quat_ = loss_quat(predicted_joint_rot, lrot_gt)
            loss_quat_v_ = loss_quat_v(predicted_joint_rot, lrot_gt)
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

def train(args, opts, model, proj, train_loader, val_loader, checkpoint=None, feature_extractor=None, backbone=None):
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
            start_time = time()
            losses = {}
            losses['loss_quat'] = AverageMeter()
            losses['loss_quat_v'] = AverageMeter()
            losses['loss_3d_pos'] = AverageMeter()
            losses['loss_3d_scale'] = AverageMeter()
            losses['loss_3d_velocity'] = AverageMeter()
            losses['total'] = AverageMeter()
            
            train_epoch(args, model=model, proj=proj, train_loader=train_loader, losses=losses, optimizer=optimizer,
                         feature_extractor=feature_extractor, backbone=backbone) 
            elapsed = (time() - start_time) / 60    

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                e1, e2, losses_test = evaluate(args, model=model, proj=proj, val_loader=val_loader,
                                               feature_extractor=feature_extractor, backbone=backbone)
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
                    wandb.log({'epoch': epoch+1, 'loss_3d_pos': losses['loss_3d_pos'.avg]})
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
                        rot=args.rot_represent)
    test_data = KinPolyDataset(dataset_path=args.data_root,
                        config_path=args.config,
                        image_tmpl=args.image_tmpl,
                        transform=img_transforms,
                        clip_length=args.clip_len,
                        mode='test',
                        use_slam=True,
                        rot=args.rot_represent)

    train_dataset, _ = random_split(train_data, [args.train_size, len(train_data) - args.train_size])
    test_dataset, _ = random_split(test_data, [args.val_size, len(test_data) - args.val_size])
    val_dataset = test_dataset
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset,
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=False)
    # ---------------------------------------------------------------------------
    # wandb Log
    run_dir = Path("/home/litianyi/workspace/EgoMotion/logs/wandb/" + args.experiment_name)
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
                "frames": int(args.clip_len / 2),
                "pose_represent": "joint_rot--" + args.rot_represent
            },
            dir=str(run_dir),
        )
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
        train(args, opts, model, proj, train_loader, val_loader, checkpoint, feature_extractor, backbone)
    else:
        e1, e2, results_all = evaluate(args, model, proj, val_loader, feature_extractor, backbone) 

def save_checkpoint(ckpt_path, epoch, lr, optimizer, model, min_loss):
    print('Saving checkpoint to', ckpt_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'backbone': model['TimesFormer'].state_dict(),
        'proj': model['proj'].state_dict(),
        'min_loss' : min_loss
    }, ckpt_path)

if __name__=='__main__':
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    main(args, opts)

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