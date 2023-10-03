import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from model.twin_network import TwinNetwork, R3D
from model.loss import *
from model.proj import Projector
from dataset.ego_dataset import EgoMotionDataset
from model.timesformer.models.vit import TimeSformer
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
from MotionBERT.lib.utils.tools import *
from functools import partial
from train_on_slam import depth_estimate
import joblib

if __name__=='__main__':
    config_path = 'config/DST_slam_train.yaml'
    args = get_config(config_path)
    checkpoints_path = "/data/newhome/litianyi/model/EgoMotion/checkpoints/exp04/best_epoch.pth"
    dataset_path = '/data/newhome/litianyi/dataset/EgoMotion/'
    pose_path = os.path.join(dataset_path, 'egomotion_pose_30fps.p')
    pose = joblib.load(pose_path)
    pose_walk = pose['02_01']['trans'][10:11] * 0.0564
    print("pose: ", pose_walk)
     # Model
    model = TimeSformer(img_size=224, num_classes=224, num_frames=8, 
                        attention_type='divided_space_time', 
                        pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
    backbone = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    # projector
    proj = Projector(224, 17*3*8, 512)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model)
        proj = nn.DataParallel(proj)
        model_backbone = model_backbone.cuda()
        proj = proj.cuda()
    checkpoint = torch.load(checkpoints_path, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['backbone'], strict=True)
    proj.load_state_dict(checkpoint['proj'], strict=True)

    img_transforms = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()])

    egodata = EgoMotionDataset(dataset_path=args.data_root,
                         config_path=args.config,
                         image_tmpl=args.image_tmpl,
                         transform=img_transforms,
                         clip_length=args.clip_len)
    # test_loader = DataLoader(egodata,
    #                           batch_size=1, shuffle=True,
    #                           num_workers=4, pin_memory=False)
    _, batch_gt, batch_input = egodata[20]
    N = 1
    C = 8
    # print("input shape: ", batch_input.shape)
    if torch.cuda.is_available():
        batch_input = torch.tensor(batch_input).cuda().reshape(-1, 3, 224, 224)
        batch_gt = torch.tensor(batch_gt).cuda()
        
    # predict depth
    with torch.no_grad():
        depth = depth_estimate(batch_input*255, feature_extractor, backbone)  ### Normalize->[0,255]
    depth = depth.reshape(N, C, 1, 224, 224)
    # Predict 3D poses
    inputs = torch.repeat_interleave(depth, 3, dim=2).permute(0, 2, 1, 3, 4)
    embeddings = model_backbone(inputs)
    # (N, F, J, C) = (32, 10, 17, 3)
    predicted_3d_pos = proj(embeddings).reshape(N, -1, 17, 3)    # (N, T, 17, 3)
    print(predicted_3d_pos.shape)
    print("predict 3d pose[0]: ", predicted_3d_pos[0][0])
    print("predict 3d pose[7]: ", predicted_3d_pos[0][7])
    print("ground truth 3d pose", batch_gt[0] * 0.0564)