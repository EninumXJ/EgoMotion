### finetune on MotionBERT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import wandb
import prettytable
from pathlib import Path
### Dataset and Model
from model.twin_network import TwinNetwork, R3D
from model.loss import *
from model.proj import Projector
from dataset.ego_dataset import EgoMotionDataset

from MotionBERT.lib.model.loss import *

import os
import numpy as np
import argparse
import random
from time import time
from tqdm import tqdm
from MotionBERT.lib.utils.tools import *
from MotionBERT.lib.utils.learning import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-pb', '--pretrained_mb', default='', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-pt', '--pretrained_twin', default='', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-pp', '--pretrained_proj', default='', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-backbone', '--backbone', default='DSTformer', type=str, help='backbone of MotionBERT')
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate(args, twin_net, proj, model, val_loader):
    print('INFO: Testing')
    results_all = []; gt_all = []
    twin_net.eval()
    proj.eval()
    model.eval()            
    with torch.no_grad():
        with tqdm(total=args.val_size) as pbar:
            for idx, (batch_input, batch_gt) in enumerate(val_loader):
                N, T = batch_gt.shape[:2]
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda().permute(0,2,1,3,4)
                
                reprentation = twin_net(batch_input)
                embeddings = proj(reprentation).reshape(N, -1, 17, 3)
                predicted_3d_pos = model(embeddings)    # (N, T, 17, 3)

                results_all.append(predicted_3d_pos.cpu().numpy())
                gt_all.append(batch_gt.cpu().numpy())
                if idx % 10 == 0:
                    pbar.update(10)
                if idx > args.val_size:
                    break
    
    gt_all = np.concatenate(gt_all)
    results_all = np.concatenate(results_all)
    print("test clips: ", results_all.shape[0])
    num_test_frames = len(results_all)
    frames = np.array(range(num_test_frames))
    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    
    final_results_1 = []; final_results_2 = []
    for idx in range(num_test_frames):
        gt = gt_all[idx]
        pred = results_all[idx]
        
        # Root-relative Errors
        pred = pred - pred[:,0:1,:]
        gt = gt - gt[:,0:1,:]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        # print("err1: ", err1)
        # print("err2: ", err2)
        e1_all[idx] += np.mean(err1)
        e2_all[idx] += np.mean(err2)
        oc[idx] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            final_results_1.append(err1)
            final_results_2.append(err2)
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name']
    
    summary_table.add_row(['P1'])
    summary_table.add_row(['P2'])
    print(summary_table)
    e1 = np.mean(np.array(final_results_1)) * 100
    e2 = np.mean(np.array(final_results_2)) * 100
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')
    return e1, e2, results_all

def train_epoch(args, model_pos, twin_net, proj, train_loader, losses, optimizer):
    model_pos.train()
    twin_net.train()
    proj.train()
    with tqdm(total=len(train_loader)) as pbar:
        for idx, (batch_input, batch_gt) in enumerate(train_loader):
            batch_size = len(batch_input)
            # print("input shape: ", batch_input.shape)
            if torch.cuda.is_available():
                batch_input = batch_input.cuda().permute(0,2,1,3,4)
                batch_gt = batch_gt.cuda()
            # Predict 3D poses
            reprentation = twin_net(batch_input)
            embeddings = proj(reprentation).reshape(batch_size, -1, 17, 3)
            # print("embedding shape: ", embeddings.shape)
            # (N, F, J, C) = (32, 10, 17, 3)
            predicted_3d_pos = model_pos(embeddings)    # (N, T, 17, 3)
            optimizer.zero_grad()

            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            loss_total = loss_3d_pos + \
                            args.lambda_scale       * loss_3d_scale + \
                            args.lambda_3d_velocity * loss_3d_velocity + \
                            args.lambda_lv          * loss_lv + \
                            args.lambda_lg          * loss_lg + \
                            args.lambda_a           * loss_a  + \
                            args.lambda_av          * loss_av
            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
            print("loss_total: ", loss_total.item())
            loss_total.backward()
            optimizer.step()
            # evaluate(model_pos, twin_net, proj, val_loader)
            if idx % 10 == 0:
                pbar.update(10)

def train_with_config(args, opts, transforms):
    print(args)
    # -------------------------------------------------------------------
    # Dataset
    print('Loading dataset...')
    egodata = EgoMotionDataset(dataset_path=args.data_root,
                         config_path=args.config,
                         image_tmpl=args.image_tmpl,
                         transform=transforms,
                         clip_length=args.clip_len)
    
    train_dataset, val_dataset = random_split(egodata, [args.train_size, len(egodata) - args.train_size])
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset,
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=False)
    # -------------------------------------------------------------------
    # wandb Log
    run_dir = Path("/home/litianyi/workspace/EgoMotion/logs/wandb/" + args.experiment_name)
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    if opts.use_wandb:
        wandb.init(
            project="FinetuneOnMotionBERT",
            name=args.experiment_name,
            job_type="training",
            # hyperparameters
            config={
                "dataset": "EgoMotion",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.learning_rate,
                "clip_length": args.clip_len,
            },
            dir=str(run_dir),
        )
    # -------------------------------------------------------------------
    # Load Checkpoint: TwinNetwork
    twin_net = R3D()
    proj = Projector(256, 51*10)
    # Load Checkpoint: MotionBERT
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        twin_net = nn.DataParallel(twin_net)
        proj = nn.DataParallel(proj)
        model_backbone = model_backbone.cuda()
        twin_net = twin_net.cuda()
        proj = proj.cuda()
    
    if opts.pretrained_mb or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.pretrained_mb
        print('Loading MotionBERT checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone
    if opts.pretrained_twin or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.pretrained_twin
        print('Loading TwinNetwork checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        twin_net.load_state_dict(checkpoint['state_dict'], strict=True)
    if opts.pretrained_proj or opts.evaluate:    
        chk_filename = opts.evaluate if opts.evaluate else opts.pretrained_proj
        print('Loading Projector checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        twin_net.load_state_dict(checkpoint['state_dict'], strict=True)
    
    # -------------------------------------------------------------------
    # Model Training Settings
    # if args.partial_train:
    model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate:        
        lr = args.learning_rate
        optimizer = optim.AdamW([{"params": twin_net.parameters()}, {"params": proj.parameters()}],
                                 lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']
        min_loss = 100000
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            N = 0
                        
            train_epoch(args, model_pos=model_pos, twin_net=twin_net, proj=proj, train_loader=train_loader, losses=losses, optimizer=optimizer) 
            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                e1, e2, results_all = evaluate(args, twin_net=twin_net, proj=proj, model=model_pos, val_loader=val_loader)
                print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg,
                    e1, e2))
                # ---------------------------------------------------------------
                # wandb log
                wandb.log({'epoch': epoch+1, 'Error P1': e1})
                wandb.log({'epoch': epoch+1, 'Error P2': e2})
                wandb.log({'epoch': epoch+1, 'loss_3d_pos': losses['3d_pos'].avg})
                wandb.log({'epoch': epoch+1, 'loss_2d_proj': losses['2d_proj'].avg})
                wandb.log({'epoch': epoch+1, 'loss_3d_scale': losses['3d_scale'].avg})
                wandb.log({'epoch': epoch+1, 'loss_3d_velocity': losses['3d_velocity'].avg})
                wandb.log({'epoch': epoch+1, 'loss_lv': losses['lv'].avg})
                wandb.log({'epoch': epoch+1, 'loss_lg': losses['lg'].avg})
                wandb.log({'epoch': epoch+1, 'loss_a': losses['angle'].avg})
                wandb.log({'epoch': epoch+1, 'loss_av': losses['angle_velocity'].avg})
                wandb.log({'epoch': epoch+1, 'loss_total': losses['total'].avg})
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
            chk_path_latest = os.path.join(chk_dir, 'latest_epoch.pth')
            chk_path_best = os.path.join(chk_dir, 'best_epoch.pth')
            model = {'twin_net': twin_net, 'proj': proj, 'backbone':model_pos}
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model, min_loss)

    if opts.evaluate:
        e1, e2, results_all = evaluate(args, model_pos, twin_net, proj, val_loader)

def save_checkpoint(ckpt_path, epoch, lr, optimizer, model, min_loss):
    print('Saving checkpoint to', ckpt_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'twin_net': model['twin_net'].state_dict(),
        'proj': model['proj'].state_dict(),
        'backbone': model['backbone'].state_dict(),
        'min_loss' : min_loss
    }, ckpt_path)

if __name__=='__main__':
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    normalize])
    train_with_config(args, opts, img_transforms)