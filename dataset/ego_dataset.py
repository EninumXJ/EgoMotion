import os
import torch
import torch.utils.data
import yaml
from glob import glob
from pathlib import Path
from PIL import Image

class EgoDataset(torch.utils.data.Dataset):
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


if __name__=='__main__':
    config_path = '../remy_2scenes.yml'
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
    ego_dataset = EgoDataset(dataset_path=dataset_path,
                             config_path=config_path,
                             image_tmpl=image_tmpl,
                             transform=transforms
                             )
    
    index = 177
    action, pair_of_clip = ego_dataset[index]
    print(action, pair_of_clip.shape)
    