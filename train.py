from model.twin_network import TwinNetwork, R3D
from dataset.ego_dataset import EgoDataset
from utils.transforms import *
from utils.utils import simp, norm
from options.twin_opt import TrainOptions
from torch.nn.utils import clip_grad_norm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import shutil
import time
from tqdm import tqdm
import wandb
import os
from pathlib import Path

def main(opt, transforms=None):
    #----------------------------------------
    # wandn init
    run_dir = Path("/data/newhome/litianyi/logs/EgoMotion")
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if opt.use_wandb:
        wandb.init(
            project="TwinNetwork",
            name=opt.name,
            job_type="training",
            # hyperparameters
            config={
                "dataset": "EgoMotion",
                "epochs": opt.n_epochs,
                "start_epoch": opt.epoch,
                "batch_size": opt.batch_size,
                "lr": opt.lr,
                "lr_steps": opt.lr_steps,
                "architecture": opt.arch,
                "clip_length": opt.clip_length,
                "modality": opt.modality,
            },
            dir=str(run_dir),
        )
    #----------------------------------------

    # model = TwinNetwork(base_model=opt.arch, modality=opt.modality, clip_length=opt.length)
    model = R3D()
    if opt.gpus == '0':
        model = model.to('cuda:0')
    else:
        model = torch.nn.DataParallel(model, device_ids=opt.gpus).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    egodata = EgoDataset(dataset_path=opt.dataroot,
                         config_path=opt.config,
                         image_tmpl=opt.image_tmpl,
                         transform=transforms)
    print("len of dataset: ", len(egodata))
    '''len of train dataset = 83871'''
    train_dataset, val_dataset = random_split(egodata, [opt.train_size, len(egodata) - opt.train_size])
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size, shuffle=True,
                              num_workers=4, pin_memory=False)

    optimizer = torch.optim.SGD(model.parameters(),
                                opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    total_iters= 0
    for epoch in range(opt.start_epoch, opt.n_epochs):
        adjust_learning_rate(optimizer, epoch, opt.lr_steps)

        # switch to train mode
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{opt.n_epochs + 1}")
        for action, clip_pair in pbar:
            # measure data loading time
            total_iters += opt.batch_size
            # clip = clip_pair.cuda().reshape(-1, opt.clip_length, 3, h, w).permute(0,2,1,3,4)
            clip_a = clip_pair.cuda()[:,0,...].permute(0,2,1,3,4)
            clip_b = clip_pair.cuda()[:,1,...].permute(0,2,1,3,4)
            # print("clip_a shape: ", clip_a.shape)
            clip_negative_1 = torch.flip(clip_a, dims=[2])
            clip_negative_2 = torch.flip(clip_a, dims=[4])    # 图像水平翻转
            input = torch.cat((clip_a, clip_b, clip_negative_1, clip_negative_2), dim=0)
            # compute output
            output = model(input)
            output_a, output_b, output_negative_1, output_negative_2 = torch.split(output, clip_a.shape[0], dim=0)
            # output_a = model(clip_a)   #(batch, 1024)
            # output_b = model(clip_b)
            # output_negative_1 = model(clip_negative_1)
            # output_negative_2 = model(clip_negative_2)

            # print("output shape: ", output_a.shape)
            # norm
            # print("d_min shape: ", d_min.shape)
            output_norm_a = norm(output_a)
            output_norm_b = norm(output_b)
            output_norm_neg_1 = norm(output_negative_1)
            output_norm_neg_2 = norm(output_negative_2)
            zeros = torch.zeros_like(output_norm_a)
            # compute the difference between two twin clips
            # loss_1 = criterion(output_norm_a, output_norm_b)
            # loss_2 = 1. / torch.norm(output_norm_a, p=1) + 1. / torch.norm(output_norm_b, p=1)
            # loss_3 = 1. / torch.norm(output_norm_a-ones, p=1) + 1. / torch.norm(output_norm_b-ones, p=1)
            # loss_4 = -criterion(output_norm_a, output_norm_neg)
            loss_1 = torch.norm(output_norm_b - output_norm_a)
            loss_2 = torch.norm(output_norm_b - output_norm_neg_1)
            loss_3 = torch.norm(output_norm_b - output_norm_neg_2)
            loss = torch.max(torch.tensor(0), loss_1 - 0.5 * loss_2 - 0.5 * loss_3 + opt.epsilon)
            # loss = loss_1 + opt.L1 * loss_2 + opt.L2 * loss_3 + opt.L3 * loss_4
            # record loss

            # compute gradient and do SGD step
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if opt.use_wandb:
                if total_iters % 200 == 0:
                    idx = np.random.randint(0, clip_pair.shape[0])
                    image_1 = wandb.Image(
                        simp(clip_pair[idx][0][5]*255).transpose(1,2,0),
                        caption="twin clip a: "+action[idx]
                    )
                    image_2 = wandb.Image(
                        simp(clip_pair[idx][1][5]*255).transpose(1,2,0),
                        caption="twin clip b: "+action[idx]
                    )
                    image_3 = wandb.Image(
                        simp(output_a[idx].reshape(16,16)*255),
                        caption="feature a: "+action[idx]
                    )
                    image_4 = wandb.Image(
                        simp(output_b[idx].reshape(16,16)*255),
                        caption="feature b: "+action[idx]
                    )
                    wandb.log({"twin clip a": image_1})
                    wandb.log({"twin clip b": image_2})
                    wandb.log({"feature a": image_3})
                    wandb.log({"feature b": image_4})

                wandb.log({'loss_1':loss_1.item()})
                wandb.log({'loss_2':loss_2.item()})
                wandb.log({'loss_3':loss_3.item()})
                # wandb.log({'loss_4':loss_4.item()})
                wandb.log({'loss':loss.item()})

        # evaluate on validation set
        if (epoch + 1) % opt.save_epoch_freq == 0 or epoch == opt.n_epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
            })

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = opt.lr * decay
    decay = opt.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * 1
        param_group['weight_decay'] = decay * 1

def save_checkpoint(state, filename='checkpoint.pth'):
    filename = '_'.join((opt.modality.lower(), filename))
    torch.save(state, os.path.join("checkpoints", opt.name, filename))

if __name__=='__main__':
    opt = TrainOptions().parse()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    normalize])
    main(opt, img_transforms)