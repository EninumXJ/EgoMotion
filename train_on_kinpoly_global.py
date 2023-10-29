import os
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
    parser.add_argument("--config", type=str, default="config/DST_slam_train.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-backbone', '--backbone', default='Timesformer', type=str, help='backbone of MotionBERT')
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    opts = parser.parse_args()
    return opts

def evaluate(args, model, proj_trans, proj_rot, val_loader, feature_extractor=None, backbone=None):
    print('INFO: Testing')
    trans_error_all = []; rot_error_all = []; losses_all = {}
    losses_all['total'] = AverageMeter()
    losses_all['loss_trans_v'] = AverageMeter()
    losses_all['loss_trans'] = AverageMeter()
    losses_all['loss_rot_v'] = AverageMeter()
    losses_all['loss_rot'] = AverageMeter()
    model.eval()
    proj_trans.eval()
    proj_rot.eval()      
    with torch.no_grad():
        with tqdm(total=args.val_size) as pbar:
            for idx, (img_clip, trans_diff, rot_diff, root_traj, _) in enumerate(val_loader):
                N, C = img_clip.shape[:2]
                if torch.cuda.is_available():
                    img_clip = img_clip.cuda()
                    trans_diff = trans_diff.cuda()  # N X (T-1) X 3
                    rot_diff = rot_diff.cuda()    # N X (T-1) X 4
                    root_traj = root_traj.cuda()  # N X T x 7
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
                predicted_trans_diff = proj_trans(embeddings).reshape(N, -1, 3)    # (N, T-1, 3)
                if args.rot_represent == 'quat4d':
                    predicted_rot_diff = proj_rot(embeddings).reshape(N, -1, 4)    # (N, T-1, 4)
                elif args.rot_represent == 'rot6d':
                    predicted_rot_diff = proj_rot(embeddings).reshape(N, -1, 6)   # (N, T-1, 6)
                    
                loss_trans_ = loss_quat(predicted_trans_diff, trans_diff)
                loss_trans_v_ = loss_quat_v(predicted_trans_diff, trans_diff)
                loss_rot6d_ = loss_quat(predicted_rot_diff, rot_diff)
                loss_rot6d_v_ = loss_quat_v(predicted_rot_diff, rot_diff)
                loss_total = args.trans_scale * loss_trans_  + \
                            args.trans_v_scale * loss_trans_v_ + \
                            args.rot_scale * loss_rot6d_ + \
                            args.rot_v_scale * loss_rot6d_v_

                losses_all['loss_trans'].update(loss_trans_.item(), N)
                losses_all['loss_trans_v'].update(loss_trans_v_.item(), N)
                losses_all['loss_rot'].update(loss_rot6d_.item(), N)
                losses_all['loss_rot_v'].update(loss_rot6d_v_.item(), N)
                losses_all['total'].update(loss_total.item(), N)
                
                trans_error = torch.cumsum(predicted_trans_diff - trans_diff, dim=1)   # 在时序维度上对轨迹的偏差进行累加
                #---------------------------------------------------
                # 计算旋转角度误差
                # 设置一个固定的单位向量，用估计得到的旋转矩阵和gt旋转矩阵分别去旋转这个向量
                # 然后计算这两个旋转过后的向量的角度之差
                if torch.cuda.is_available():
                    direction = torch.tensor([[1., 0., 0.]]).cuda().float()   # 1 X 3
                direction = direction.reshape(1, 1, 1, 3)
                if args.rot_represent == 'quat4d':
                    predicted_matrix_diff = transforms.quaternion_to_matrix(predicted_rot_diff)    # (N, T-1, 3, 3)
                    rot_matrix_diff_gt = transforms.quaternion_to_matrix(rot_diff)     # (N, T-1, 3, 3)
                elif args.rot_represent == 'rot6d':
                    predicted_matrix_diff = transforms.rotation_6d_to_matrix(predicted_rot_diff)   # (N, T-1, 3, 3)
                    rot_matrix_diff_gt = transforms.rotation_6d_to_matrix(rot_diff)     # (N, T-1, 3, 3)
                direction_predict = torch.matmul(direction, predicted_matrix_diff)    # (N, T-1, 1, 3)
                direction_gt = torch.matmul(direction, rot_matrix_diff_gt)    # (N, T-1, 1, 3)
                rot_error = torch.arccos(torch.matmul(direction_predict, direction_gt.permute(0, 1, 3, 2)).squeeze(-1)) * 180. / torch.pi   # (N, T-1, 1, 1) rad -> degree
                # print("rot_error shape: ", rot_error.shape)
                trans_error_all.append(trans_error.cpu().numpy())
                rot_error_all.append(rot_error.cpu().numpy())
                if idx % 10 == 0:
                    pbar.update(10)
                # if idx > args.val_size:
                #     break
    
    trans_errors = np.concatenate(trans_error_all)
    rot_errors = np.concatenate(rot_error_all)
    print("test clips: ", trans_errors.shape[0])
    num_test_clips = len(trans_errors)
    e1_all = np.zeros(num_test_clips)
    e2_all = np.zeros(num_test_clips)
    oc = np.zeros(num_test_clips)
    
    final_results_1 = []; final_results_2 = []
    for idx in range(num_test_clips):
        trans_err = trans_errors[idx]
        rot_err = rot_errors[idx]
        # print("trans_err shape: ", trans_err.shape)
        # print("rot_err shape: ", rot_err.shape)
        # print(rot_err)
        e1_all[idx] = np.mean(trans_err)
        e2_all[idx] = np.mean(rot_err)
        oc[idx] += 1
        final_results_1.append(e1_all[idx])
        final_results_2.append(e2_all[idx])
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name']
    
    summary_table.add_row(['P1'])
    summary_table.add_row(['P2'])
    print(summary_table)
    e1 = np.mean(np.array(final_results_1)) * 1000
    e2 = np.mean(np.array(final_results_2))
    print('Translation Error (mm):', e1, 'mm')
    print('Rotation Error (degrees):', e2, 'degrees')
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

def train_epoch(args, model, proj_trans, proj_rot, train_loader, losses, optimizer, feature_extractor=None, backbone=None):
    model.train()
    proj_trans.train()
    proj_rot.train()
    with tqdm(total=len(train_loader)) as pbar:
        for idx, (img_clip, trans_diff, rot_diff, _) in enumerate(train_loader):
            N, C = img_clip.shape[0:2]
            # print("input shape: ", batch_input.shape)
            if torch.cuda.is_available():
                img_clip = img_clip.cuda()
                trans_diff = trans_diff.cuda()  # (T-1) X 3
                rot_diff = rot_diff.cuda()    # (T-1) X 4
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
            predicted_trans_diff = proj_trans(embeddings).reshape(N, -1, 3)    # (N, T-1, 3)
            if args.rot_represent == 'quat4d':
                predicted_rot_diff = proj_rot(embeddings).reshape(N, -1, 4)    # (N, T-1, 4)
            elif args.rot_represent == 'rot6d':
                predicted_rot_diff = proj_rot(embeddings).reshape(N, -1, 6)   # (N, T-1, 6)
            optimizer.zero_grad()
            with torch.no_grad():
                pass
            
            loss_trans_ = loss_quat(predicted_trans_diff, trans_diff)
            loss_trans_v_ = loss_quat_v(predicted_trans_diff, trans_diff)
            loss_rot6d_ = loss_quat(predicted_rot_diff, rot_diff)
            loss_rot6d_v_ = loss_quat_v(predicted_rot_diff, rot_diff)
            loss_total = args.trans_scale * loss_trans_  + \
                         args.trans_v_scale * loss_trans_v_ + \
                         args.rot_scale * loss_rot6d_ + \
                         args.rot_v_scale * loss_rot6d_v_

            losses['loss_trans'].update(loss_trans_.item(), N)
            losses['loss_trans_v'].update(loss_trans_v_.item(), N)
            losses['loss_rot'].update(loss_rot6d_.item(), N)
            losses['loss_rot_v'].update(loss_rot6d_v_.item(), N)
            losses['total'].update(loss_total.item(), N)
            print("loss_total: ", loss_total.item())
            loss_total.backward()
            optimizer.step()
            # evaluate(model_pos, twin_net, proj, val_loader)
            if idx % 10 == 0:
                pbar.update(10)

def train(args, opts, model, proj_trans, proj_rot, train_loader, val_loader, checkpoint=None, feature_extractor=None, backbone=None):
    st = 0
    min_loss = 100000
    if not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW([{"params": model.parameters()}, {"params": proj_trans.parameters()}, {"params": proj_rot.parameters()}],
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
            losses['loss_rot'] = AverageMeter()
            losses['loss_rot_v'] = AverageMeter()
            losses['loss_trans'] = AverageMeter()
            losses['loss_trans_v'] = AverageMeter()
            losses['total'] = AverageMeter()
            
            train_epoch(args, model=model, proj_trans=proj_trans, proj_rot=proj_rot, train_loader=train_loader,
                         losses=losses, optimizer=optimizer, feature_extractor=feature_extractor, backbone=backbone) 
            elapsed = (time() - start_time) / 60    

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                e1, e2, losses_test = evaluate(args, model=model, proj_trans=proj_trans, proj_rot=proj_rot, val_loader=val_loader,
                                               feature_extractor=feature_extractor, backbone=backbone)
                # ---------------------------------------------------------------
                # wandb log
                if opts.use_wandb:
                    wandb.log({'epoch': epoch+1, 'Error P1': e1})
                    wandb.log({'epoch': epoch+1, 'Error P2': e2})
                    wandb.log({'epoch': epoch+1, 'loss_rot': losses['loss_rot'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_rot_v': losses['loss_rot_v'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_trans': losses['loss_trans'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_trans_v': losses['loss_trans_v'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_total_train': losses['total'].avg})
                    wandb.log({'epoch': epoch+1, 'loss_total_test': losses_test['total'].avg})
            # ------------------------------------------------------------------- 
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # -------------------------------------------------------------------
            # Save checkpoints
            chk_dir = os.path.join(opts.checkpoint, args.experiment_name)
            if not os.path.exists(chk_dir):
                os.makedirs(chk_dir)
            chk_path = os.path.join(chk_dir, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(chk_dir, 'latest_epoch_global.pth')
            chk_path_best = os.path.join(chk_dir, 'best_epoch_global.pth')
            models = {'TimesFormer': model, 'proj_trans': proj_trans, 'proj_rot': proj_rot}
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
                        coordinate='global')
    test_data = KinPolyDataset(dataset_path=args.data_root,
                        config_path=args.config,
                        image_tmpl=args.image_tmpl,
                        transform=img_transforms,
                        clip_length=args.clip_len,
                        mode='test',
                        use_slam=True,
                        rot=args.rot_represent,
                        coordinate='global')

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
                "pose_represent": "joint_rot--" + args.rot_represent,
                "coordinate": 'global',
            },
            dir=str(run_dir),
        )
    # --------------------------------------------------------------------------
    # Model
    model = TimeSformer(img_size=224, num_classes=224, num_frames=8, attention_type='divided_space_time',
                        pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    
    # projector
    proj_trans = Projector(224, 3*7, 128, 32)
    if args.rot_represent == 'rot6d':
        proj_rot = Projector(224, 6*7, 128, 64)
    elif args.rot_represent == 'quat4d':
        proj_rot = Projector(224, 4*7, 128, 64)
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
        proj_trans = nn.DataParallel(proj_trans)
        proj_trans = proj_trans.cuda()
        proj_rot = nn.DataParallel(proj_rot)
        proj_rot = proj_rot.cuda()

    checkpoint = None
    # ---------------------------------------------------------------------------
    # TODO resume from checkpoint
    if opts.resume:
        chk_filename = opts.resume
        print("Loading resume", chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['backbone'], strict=True)
        proj_trans.load_state_dict(checkpoint['proj_trans'], strict=True)
        proj_rot.load_state_dict(checkpoint['proj_rot'], strict=True)
    #----------------------------------------------------------------------------
    if not opts.evaluate:
        train(args, opts, model, proj_trans, proj_rot, train_loader, val_loader, checkpoint, feature_extractor, backbone)
    else:
        e1, e2, results_all = evaluate(args, model, proj_trans, proj_rot, val_loader, feature_extractor, backbone) 

def save_checkpoint(ckpt_path, epoch, lr, optimizer, model, min_loss):
    print('Saving checkpoint to', ckpt_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'backbone': model['TimesFormer'].state_dict(),
        'proj_trans': model['proj_trans'].state_dict(),
        'proj_rot': model['proj_rot'].state_dict(),
        'min_loss' : min_loss
    }, ckpt_path)

if __name__=='__main__':
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    main(args, opts)