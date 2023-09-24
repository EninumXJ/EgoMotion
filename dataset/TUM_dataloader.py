import os
import torch
import torch.utils.data
import yaml
from glob import glob
from pathlib import Path
from PIL import Image
from associate import associate, read_file_list
import numpy as np
from torchvision import transforms

class TUMDataset():
    def __init__(self, data_root_path, sub_name, clip_length, mode, transform):
        self.data_root_path = data_root_path
        self.sub_name = sub_name
        self.clip_length = clip_length
        self.mode = mode
        self.dataset_path = os.path.join(self.data_root_path, self.sub_name)
        dirs_list = os.listdir(self.dataset_path)
        self.transforms = transform
        self.data_dict = {}
        for dir in dirs_list:
            if dir == 'rgbd_dataset_freiburg1_360':
                print("dir: ", dir)
                seq_path = os.path.join(self.dataset_path, dir)
                rgb_txt_path = os.path.join(seq_path, 'rgb.txt')
                rgb_file = read_file_list(rgb_txt_path)
                depth_txt_path = os.path.join(seq_path, 'depth.txt')
                depth_file = read_file_list(depth_txt_path)
                gt_txt_path = os.path.join(seq_path, 'groundtruth.txt')
                gt_file = read_file_list(gt_txt_path)

                rgb_depth_match = associate(rgb_file, depth_file, 0, 0.02)
                rgb_gt_match = associate(rgb_file, gt_file, 0, 0.02)
                self.data_dict[dir] = {'rgb_depth_match': rgb_depth_match, 'rgb_gt_match': rgb_gt_match}

    def sample(self, seq_name, idx):
        seq_path = os.path.join(self.dataset_path, seq_name)
        rgb_txt_path = os.path.join(seq_path, 'rgb.txt')
        rgb_file = read_file_list(rgb_txt_path)
        depth_txt_path = os.path.join(seq_path, 'depth.txt')
        depth_file = read_file_list(depth_txt_path)
        gt_txt_path = os.path.join(seq_path, 'groundtruth.txt')
        gt_file = read_file_list(gt_txt_path)
        rgb_images = []
        depth_images = []
        gt = []
        for i in range(idx, idx+self.clip_length):
            # print(self.data_dict[seq_name]['rgb_depth_match'])
            rgb_path = os.path.join(seq_path, rgb_file[self.data_dict[seq_name]['rgb_depth_match'][i][0]][0])
            depth_path = os.path.join(seq_path, depth_file[self.data_dict[seq_name]['rgb_depth_match'][i][1]][0])
            rgb_images.append(self.transforms(Image.open(rgb_path)))
            depth_map = self.transforms(Image.open(depth_path))
            depth_map = depth_map / 256.0 * 65536
            depth_images.append(depth_map)
            gt.append(torch.tensor([float(i) for i in gt_file[self.data_dict[seq_name]['rgb_gt_match'][i][1]]]))
        rgb_clip = torch.stack(rgb_images, 0)
        depth_clip = torch.stack(depth_images, 0)
        gt_clip = torch.stack(gt, 0)
        return rgb_clip, depth_clip, gt_clip
 
    def generate_rand(self,):
        pass

    def generate_batch(self, batch_size):
        pass


if __name__ == '__main__':
    data_root_path = '../data/TUM'
    sub_name = 'Handheld_SLAM'
    clip_length = 30
    transform = transforms.ToTensor()
    dataset = TUMDataset(data_root_path=data_root_path, sub_name=sub_name, clip_length=clip_length, mode='train', transform=transform)
    rgb, depth, gt = dataset.sample('rgbd_dataset_freiburg1_360', 50)
    print(gt.shape)