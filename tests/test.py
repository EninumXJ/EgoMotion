import sys
sys.path.append("/home/litianyi/workspace/EgoMotion")
from model.twin_network import TwinNetwork, R3D
from dataset.ego_dataset import EgoDataset
from torch import tensor
from utils.transforms import *
from utils.utils import simp, norm
from options.test_opt import TestOptions
from torch.nn.utils import clip_grad_norm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import shutil
import time
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test(opt, transforms=None):

    model = R3D()
    if opt.gpus == '0':
        model = model.to('cuda:0')
    else:
        model = torch.nn.DataParallel(model, device_ids=opt.gpus).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if opt.resume:
        print(("=> loading checkpoint '{}'".format(opt.resume)))
        checkpoint = torch.load(opt.resume)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'],strict=False)

    # model = TwinNetwork(base_model=opt.arch, modality=opt.modality, clip_length=opt.length)
    if not opt.use_feature:
        egodata = EgoDataset(dataset_path=opt.dataroot,
                            config_path=opt.config,
                            image_tmpl=opt.image_tmpl,
                            transform=transforms)
        print("len of dataset: ", len(egodata))
        '''len of train dataset = 83871'''
        train_dataset, val_dataset = random_split(egodata, [len(egodata)-opt.val_size, opt.val_size])
        val_loader = DataLoader(val_dataset,
                                batch_size=opt.batch_size, shuffle=True,
                                num_workers=4, pin_memory=False)

        total_iters= 0
        record=[];feature=[];label=[]
        # switch to eval mode
        model.eval()
        pbar = tqdm(val_loader)
        with torch.no_grad():
            for action, ind, clip_pair in pbar:
                # measure data loading time
                # print(action)
                total_iters += opt.batch_size
                w = clip_pair.shape[-2]
                h = clip_pair.shape[-1]
                # clip = clip_pair.cuda().reshape(-1, opt.clip_length, 3, w, h).permute(0,2,1,3,4)
                clip_a = clip_pair.cuda()[:,0,...].permute(0,2,1,3,4)
                clip_b = clip_pair.cuda()[:,1,...].permute(0,2,1,3,4)
                # print("clip_a shape: ", clip_a.shape)
                clip_negative = torch.flip(clip_a, dims=[2])
                # compute output
                output_a = model(clip_a)   #(batch, 1024)
                output_b = model(clip_b)
                output_negative = model(clip_negative)
                # print("output shape: ", output_a.shape)
                # norm
                # print("d_min shape: ", d_min.shape)
                output_norm_a = norm(output_a)
                output_norm_b = norm(output_b)
                output_norm_neg = norm(output_negative)
                feature.append(output_norm_a[0])  
                label.append(action[0]+':'+str(ind[0].item())) 
                record.append([action[0], ind[0].item(), output_norm_a[0], output_norm_b[0], output_norm_neg[0]])
        record_pd=pd.DataFrame(data=record,index=None,columns=['action', 'index', 'output_norm_a','output_norm_b',"output_norm_neg"])
        csv_path = os.path.join(opt.log_dir, opt.name, 'feature_record.csv')
        record_pd.to_csv(csv_path)
    
    if opt.use_feature:
        csv_path = opt.feature_path
        record_pd = pd.read_csv(opt.feature_path)
        record = record_pd.values.tolist()
    
    features = torch.stack(feature, 0).cpu().numpy()
    print("feature shape: ", features.shape)

    tsne = TSNE(n_components=3)
    tsne.fit_transform(features)
    embedding = tsne.embedding_
    print(embedding.shape)
    x=embedding[:,0]
    y=embedding[:,1]
    z=embedding[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # point_range = range(0, points.shape[0], skip) # skip points to prevent crash
    ax.scatter(x,   # x
               y,   # y
               z,   # z
            # c=points[point_range, 2], # height data for color
            cmap='spectral',
            marker="x")
    ax.axis('scaled')  # {equal, scaled}
    # for i in range(len(x)):
    #     ax.text(x[i],y[i],z[i],label[i])
    plt.savefig("tests/test.jpg")


def test_plot(opt, transforms=None):
    clip_length = 10
    image_tmpl = '{:04d}.jpg'
    dataset_path = '/data/newhome/litianyi/dataset/EgoMotion/lab/'
    # action_list = {'02_01_walk': (2,48), '02_04_jump': (3,120), '02_06_bend': (3,95),
    #                '13_04_sit_on_stepstool': (3,66), '143_19_sit_and_getup': (5,84),
    #                '16_17_turn_left': (5,69), '16_19_turn_right': (5,56)}
    action_list = {'02_02_walk':(3,48), '02_04_jump': (3,120), '26_10_bend':(5,15),
                   '13_06_sit_on_stepstool':(4,82), '75_19_medium_sit':(5,24),
                   '16_18_turn_left':(5,72), '16_20_turn_right':(4,52)}
    clip_list = []
    label = ['walk', 'jump', 'bend', 'sit_down', 'sit_up', 'turn_left', 'turn_right']
    num = [range(0,3), range(3,6), range(6,11), range(11,15), range(15,20), range(20,25), range(25,29)]
    for action, tup in action_list.items():
        for i in range(1, tup[0]+1):
            image_path = os.path.join(dataset_path, action, str(i))
            images = []
            for i in range(clip_length):
                images.append(transforms(Image.open(os.path.join(image_path, image_tmpl.format(i + tup[1]))).convert('RGB')))
            clip = torch.stack(images, 0)
            clip_list.append(clip)
    mini_batch = torch.stack(clip_list, 0)
    print(mini_batch.shape)

    model = R3D()
    if opt.gpus == '0':
        model = model.to('cuda:0')
    else:
        model = torch.nn.DataParallel(model, device_ids=opt.gpus).cuda()

    if opt.resume:
        print(("=> loading checkpoint '{}'".format(opt.resume)))
        checkpoint = torch.load(opt.resume)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'],strict=False)
    
    feature = model(mini_batch.cuda().permute(0,2,1,3,4)).detach().cpu().numpy()
    # tsne = TSNE(n_components=128, perplexity=5, method='exact')
    tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=0)
    embedding = tsne.fit_transform(feature)
    plot_embedding(embedding, label, num, 'feature embedding')
    # plot_images(embedding[0], 'walk')
    # plot_images(embedding[2], 'jump')
    # plot_images(embedding[8], 'sit_down')
    # plot_images(embedding[11], 'sit_up')
    # plot_images(embedding[16], 'turn_left')
    # plot_images(embedding[21], 'turn_right')

def plot_images(y, name):
    x = np.arange(0, 1024, 1)
    plt.plot(x, y, "yo-")
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig('tests/'+name+".png")
  
def plot_embedding(data, label, num_list, title):
    print(data.shape)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax=plt.gca()  #gca:get current axis得到当前轴
    #设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    for i in range(data.shape[0]):
        ind_bool = [i in j for j in num_list]
        # ind表示该index属于第ind个动作视频对
        ind = ind_bool.index(True)
        label_ = label[ind]
        plt.text(data[i, 0], data[i, 1], str(label_),
                 color=plt.cm.Set1(ind / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig('tests/'+"result6.png")

if __name__=='__main__':
    opt = TestOptions().parse()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    normalize])
    # test(opt, img_transforms)
    test_plot(opt, img_transforms)