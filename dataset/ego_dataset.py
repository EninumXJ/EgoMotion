import os
import torch
import torch.utils.data
import yaml
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
import joblib
import sys
sys.path.append('..')
from util.bvh2pose import Bone_Addr_17joints, Switch_Position

class TwinDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, config_path, image_tmpl, transform=None, mode='train', clip_length=10):
        self.dataset_path = dataset_path
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(),Loader=yaml.FullLoader)
        if mode == 'train':
            self.action_list = config['train']
        else:
            self.action_list = config['test']
        self.transform = transform
        self.clip_length = clip_length
        self.image_tmpl = image_tmpl
        self.preprocess(config)


    def preprocess(self, config):
        # length_list: 建立了索引到动作视频对的映射
        # pair_list: 动作-视频对
        self.length_list = []; self.pair_list = []
        len = 0; temp_len = 0
        for action in self.action_list:
            for pairs in config["video frames"][action]["lab"]:
                for key, value in pairs.items():
                    pair = value
                    temp_len = key
                self.pair_list.append({action: pair})
                # 一次取连续的10帧图像,所以要裁掉末尾
                self.length_list.append(range(len,len+temp_len-self.clip_length+1))
                len += temp_len-self.clip_length+1

    def _load_image(self, directory, index):
        return self.transform(Image.open(os.path.join(directory, self.image_tmpl.format(index))).convert('RGB'))

    def _load_clip_pair(self, index):
        ind_bool = [index in i for i in self.length_list]
        print("length_list: ", self.length_list)
        # ind表示该index属于第ind个动作视频对
        ind = ind_bool.index(True)
        pairs = self.pair_list[ind]
        for i in pairs.keys():
            action = i
        pair = pairs[action]
        idx = index - self.length_list[ind][0]
        path_1 = os.path.join(action, str(pair[0]))
        path_2 = os.path.join(action, str(pair[1]))
        clip_1 = self._load_clip(path_1, idx)
        clip_2 = self._load_clip(path_2, idx)
        return action, torch.stack((clip_1, clip_2), dim=0)

    def _load_clip(self, path, index):
        video_directory = os.path.join(self.dataset_path, 'lab', path)
        images = []
        for i in range(index, index+self.clip_length):
            images.append(self._load_image(video_directory, i))
        clip = torch.stack(images, 0)
        return clip

    def __len__(self):
        return self.length_list[-1][1]

    def __getitem__(self, idx):
        action, clip_pair = self._load_clip_pair(idx)
        return action, clip_pair

class EgoMotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, config_path, image_tmpl="{:04d}.jpg", transform=None, mode='train', clip_length=10, use_slam=True):
        self.dataset_path = dataset_path
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(),Loader=yaml.FullLoader)
        if mode == 'train':
            self.action_list = config['train']
        else:
            self.action_list = config['test']
        self.transform = transform
        self.clip_length = clip_length
        self.image_tmpl = image_tmpl
        pose_path = os.path.join(self.dataset_path, 'egomotion_pose_30fps.p')
        # pose_path = os.path.join(self.dataset_path, 'egomotion_pose.p')
        # if use_slam then input slam results instead of images
        self.use_slam = use_slam

        self.read_bodypose(pose_path)
        self.preprocess(config)
     
    def read_bodypose(self, pose_path):
        self.pose = joblib.load(pose_path)

    def preprocess(self, config):
        # length_list: 建立了索引到mocap数据的映射
        # pair_list: 动作-视频对
        self.action_index_range = []
        self.action_video_num = {}
        num_frames = 0; num = 0
        for action in config["video_frames"]:
            # action: '02_01_walk'
            if config["video_frames"][action] == None:
                continue
            # self.action_video_num[action] = {}
            for index in config["video_frames"][action]["lab"]:
                # index: '1', '2', ...
                video_frames = int(config["video_frames"][action]["lab"][index]) + 1  # important
                act1, act2, _ = action.split("_",2)
                act = act1 + '_' + act2
                # print("act: ", act)
                mocap_frames = self.pose[act]['trans'].shape[0]
                # mocap_frames_2 = int(config["mocap_frames"][action])
                # print("mocap_frames: ", mocap_frames)
                # print("mocap_frames_2: ", mocap_frames_2)
                frames = min(video_frames, mocap_frames) - 2*self.clip_length  # 去掉首尾2个clip
                self.action_video_num[num] = (action, index)
                self.action_index_range.append(range(num_frames, num_frames+frames))
                num += 1
                num_frames += frames
        # print("action_index_range: ", self.action_index_range)
        # print("action_video_num: ", self.action_video_num)

    def _load_image(self, directory, index):
        # print(self.image_tmpl)
        # print(self.image_tmpl.format(index))
        return self.transform(Image.open(os.path.join(directory, self.image_tmpl.format(index))).convert('RGB'))
    
    def _load_image_clip(self, directory, index):
        images = []
        for i in range(index, index+self.clip_length):
            images.append(self._load_image(directory, i))
        clip = torch.stack(images, 0)
        return clip

    def _load_mocap(self, action, index):
        ### h36m pose: 17 joints
        act1, act2, _ = action.split("_",2)
        act = act1 + '_' + act2
        # pose_former = self.pose[act]['trans'][index-1:index+self.clip_length-1]
        # pose_present = self.pose[act]['trans'][index:index+self.clip_length]
        # pose_gt = torch.tensor(pose_present - pose_former)
        # print("pose_former: ", pose_former)
        # print("pose_present: ", pose_present)
        # print("pose_gt shape: ", pose_gt.shape)
        pose = self.pose[act]['trans'][index:index+self.clip_length]
        # print("pose shape: ", pose.shape)
        pose = pose[:, Bone_Addr_17joints, :]
        pose[:, 9, ...] = (pose[:, 11, ...] + pose[:, 12, ...]) / 2  ### compute chin position
        # print("pose shape: ", pose.shape)
        pose_gt = pose[:, Switch_Position, :]
        pose_gt = pose_gt - pose_gt[0:1, 0:1, :]    ### 减去t0时刻root的坐标
        return pose_gt

    def _load_slam_results(self, directory, index):
        slam_results = torch.tensor(np.load(os.path.join(directory, 'feature_10frames.npy')))
        return slam_results[index:index+self.clip_length] 

    def sample(self, input, mocap, ratio=0.5):
        new_num_frames = int(ratio * self.clip_length)
        downsamp_inds = np.linspace(0, self.clip_length-1, num=new_num_frames, endpoint=False, dtype=int)
        input_sample = input[downsamp_inds]
        mocap_sample = mocap[downsamp_inds]
        return input_sample, mocap_sample

    def __len__(self):
        return self.action_index_range[-1][-1]

    def __getitem__(self, index):
        ### load img
        ind_bool = [index in i for i in self.action_index_range]
        # ind表示该index属于第ind个动作视频对
        ind = ind_bool.index(True)
        action, sub_dir = self.action_video_num[ind]
        # print("action: ", action)
        # print("sub_dir: ", sub_dir)
        video_path = os.path.join(self.dataset_path, 'lab', action, str(sub_dir))
        slam_results_path = os.path.join(self.dataset_path, 'features', action, 'lab', str(sub_dir))
        index_in_video = index - self.action_index_range[ind][0] + self.clip_length
        index_in_mocap = index_in_video
        pose_gt = self._load_mocap(action, index_in_mocap)
        if self.use_slam:
            slam_features = self._load_slam_results(slam_results_path, index_in_video)
            slam_clip, pose_gt = self.sample(slam_features, pose_gt)
            return slam_clip, pose_gt
        else:
            img_clip = self._load_image_clip(video_path, index_in_video)
            ### load mocap data
            img_clip, pose_gt = self.sample(img_clip, pose_gt)
            return img_clip, pose_gt

if __name__=='__main__':
    config_path = '../data/EgoMotion/meta_remy.yml'
    # with open(config_path, encoding="utf-8") as f:
    #     config = yaml.load(f,Loader=yaml.FullLoader)
    # action_list = config['train']
    # length_list = []; pair_list = []
    # for action in action_list:
    #         for pairs in config["video frames"][action]["lab"]:
    #             print(pairs.keys())
    dataset_path = '/data/newhome/litianyi/dataset/EgoMotion/'
    image_tmpl = "{:04d}.jpg"
    import torchvision
    import numpy as np
    class Scale():
        def __init__(self, size, interpolation=Image.BILINEAR):
            self.worker = torchvision.transforms.Resize(size, interpolation)

        def __call__(self, img):
            return self.worker(img)
        
    class ToTorchFormatTensor(object):
        """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
        to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
        def __init__(self, div=True):
            self.div = div

        def __call__(self, pic):
            if isinstance(pic, np.ndarray):
                # handle numpy array
                img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
            else:
                # handle PIL Image
                img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
                img = img.view(pic.size[1], pic.size[0], len(pic.mode))
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
            return img.float().div(255) if self.div else img.float()
    
    class GroupNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
            rep_std = self.std * (tensor.size()[0]//len(self.std))

            # TODO: make efficient
            for t, m, s in zip(tensor, rep_mean, rep_std):
                t.sub_(m).div_(s)

            return tensor

    transforms = torchvision.transforms.Compose([
                                            Scale(256),
                                            ToTorchFormatTensor(),
                                            GroupNormalize(
                                                mean=[.485, .456, .406],
                                                std=[.229, .224, .225])
                                            ])
    ego_dataset = EgoMotionDataset(dataset_path=dataset_path,
                             config_path=config_path,
                             image_tmpl=image_tmpl,
                             transform=transforms,
                             clip_length=60)
    print(len(ego_dataset))
    index = 177
    slam, pose_gt = ego_dataset[index]
    print(pose_gt.shape)
    print(slam.shape)
    # print(action, pair_of_clip.shape)
    