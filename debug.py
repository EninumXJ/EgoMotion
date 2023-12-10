import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from model.twin_network import TwinNetwork, R3D
from model.loss import *
from model.proj import Projector
from dataset.ego_dataset import EgoMotionDataset
from dataset.kinpoly_dataset import KinPolyDataset
from model.timesformer.models.vit import TimeSformer
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
from MotionBERT.lib.utils.tools import *
from functools import partial
from train_on_egomotion import depth_estimate
from utils.pose2bvh import write_standard_bvh
import joblib

def debugTimeSformer(resume):
    model = TimeSformer(img_size=224, num_classes=224, num_frames=8, 
                        attention_type='divided_space_time', 
                        pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    print("model: ", model)
    proj = Projector(224, 23*6*7, 512, 1024)
    config = "/data/newhome/litianyi/dataset/kin_poly/MoCap_dataset/mocap_meta.yml"
    data_root = "/data/newhome/litianyi/dataset/kin_poly/MoCap_dataset/"
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        proj = nn.DataParallel(proj)
        proj = proj.cuda()
    chk_filename = resume
    print("Loading resume", chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['backbone'], strict=True)
    proj.load_state_dict(checkpoint['proj'], strict=True)

    img_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224)),
                    torchvision.transforms.ToTensor()])
    
    test_data = KinPolyDataset(dataset_path=data_root,
                        config_path=config,
                        image_tmpl="{:05d}.jpg",
                        transform=img_transforms,
                        clip_length=16,
                        mode='train',
                        use_slam=False,
                        rot='rot6d')
    lrot_gt, img_clip, root, joint_rot = test_data[0]
    if torch.cuda.is_available():
        img_clip = img_clip.cuda().unsqueeze(0)
        lrot_gt = lrot_gt.cuda().unsqueeze(0)[:, 1:]
        joint_rot = joint_rot.cuda().unsqueeze(0)
        root = root.cuda().unsqueeze(0)
    inputs = img_clip.permute(0, 2, 1, 3, 4)
    embeddings = model(inputs)
    predicted_joint_rot = proj(embeddings).reshape(1, -1, 23, 6)
    print("output: ", predicted_joint_rot)
    print("ground truth: ", lrot_gt)


if __name__=='__main__':
    # resume = "/data/newhome/litianyi/model/KinPoly/checkpoints/exp03/best_epoch.pth"
    # #debugTimeSformer(resume)
    # proj2 = nn.RNN(224, 24*6, 7)
    # proj1 = 
    # bs = 32
    # input = torch.ones((bs, 3, 8, 224, 224))
    # model = TimeSformer(img_size=224, num_classes=224, num_frames=8, 
    #                     attention_type='divided_space_time', 
    #                     pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    # embedding = model(input)
    # print("embedding shape: ", embedding.shape)
    # output, _ = proj(embedding)
    # print("output shape: ", output.shape)

    import torch
    from model.timesformer.models.vit import TimeSformer

    model = TimeSformer(img_size=224, num_classes=512, num_frames=16, attention_type='divided_space_time',
                        pretrained_model="/data/newhome/litianyi/model/TimeSformer/jx_vit_base_p16_224-80ecf9dd.pth")
    print("model: ", model)