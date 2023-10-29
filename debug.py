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
    resume = "/data/newhome/litianyi/model/KinPoly/checkpoints/exp03/best_epoch.pth"
    debugTimeSformer(resume)
    # config_path = 'config/DST_slam_train.yaml'
    # args = get_config(config_path)
    # checkpoints_path = "/data/newhome/litianyi/model/EgoMotion/checkpoints/exp04/best_epoch.pth"
    # dataset_path = '/data/newhome/litianyi/dataset/EgoMotion/'
    # pose_path = os.path.join(dataset_path, 'egomotion_pose_30fps.p')
    # pose = joblib.load(pose_path)
    # pose_walk = pose['02_01']['trans'][10:11] * 0.0564
    # print("pose: ", pose_walk)
    #  # Model
    # model = TimeSformer(img_size=224, num_classes=224, num_frames=8, 
    #                     attention_type='divided_space_time', 
    #                     pretrained_model='/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    # feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
    # backbone = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    # # projector
    # proj = Projector(224, 17*3*8, 512)
    # if torch.cuda.is_available():
    #     model_backbone = nn.DataParallel(model)
    #     proj = nn.DataParallel(proj)
    #     model_backbone = model_backbone.cuda()
    #     proj = proj.cuda()
    # checkpoint = torch.load(checkpoints_path, map_location=lambda storage, loc: storage)
    # model_backbone.load_state_dict(checkpoint['backbone'], strict=True)
    # proj.load_state_dict(checkpoint['proj'], strict=True)

    # img_transforms = transforms.Compose([
    #                 transforms.Resize((224,224)),
    #                 transforms.ToTensor()])

    # egodata = EgoMotionDataset(dataset_path=args.data_root,
    #                      config_path=args.config,
    #                      image_tmpl=args.image_tmpl,
    #                      transform=img_transforms,
    #                      clip_length=args.clip_len)
    # # test_loader = DataLoader(egodata,
    # #                           batch_size=1, shuffle=True,
    # #                           num_workers=4, pin_memory=False)
    # _, batch_gt, batch_input = egodata[20]
    # N = 1
    # C = 8
    # # print("input shape: ", batch_input.shape)
    # if torch.cuda.is_available():
    #     batch_input = torch.tensor(batch_input).cuda().reshape(-1, 3, 224, 224)
    #     batch_gt = torch.tensor(batch_gt).cuda()
        
    # - predict depth ------------------------------------------------------------------------
    # with torch.no_grad():
    #     depth = depth_estimate(batch_input*255, feature_extractor, backbone)  ### Normalize->[0,255]
    #     depth = depth.reshape(N, C, 1, 224, 224)
    #     print("depth: ", depth[0][0][0])
    #     # Predict 3D poses
    #     inputs = torch.repeat_interleave(depth, 3, dim=2).permute(0, 2, 1, 3, 4)
    #     embeddings = model_backbone(inputs)
    #     # (N, F, J, C) = (32, 10, 17, 3)
    #     predicted_3d_pos = proj(embeddings).reshape(N, -1, 17, 3)    # (N, T, 17, 3)
    #     # print(predicted_3d_pos.shape)
    #     # print("predict 3d pose[0]: ", predicted_3d_pos[0][0])
    #     # print("predict 3d pose[7]: ", predicted_3d_pos[0][7])
    #     # print("ground truth 3d pose", batch_gt[0])
    #     predicted_3d_pos_0 = predicted_3d_pos[0].cpu().numpy()
    #     print("3d pos shape: ", predicted_3d_pos_0.shape)
    #     index = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10,
    #              11, 12, 13, 14, 15, 16]
    #     predicted_3d_pos_0 = predicted_3d_pos_0[:, index, :]
    #     file_path = "predict_3d_pos.bvh"
    #     write_standard_bvh(file_path, predicted_3d_pos_0)

    # -------------------------------------------------------------------------
    ### ground truth debug
    # motion_path = "data/EgoMotion/egomotion_pose_30fps.p"
    # motion = joblib.load(motion_path)
    # pose_gt = motion['02_01']['trans'][0:8]
    # index1 = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20]
    # pose_ = pose_gt[:, index1, :]
    # pose_[:, 9, ...] = (pose_[:, 11, ...] + pose_[:, 12, ...]) / 2
    # index2 = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # pose = pose_[:, index2, :]
    # # print("pose: ", pose)
    # print("pose_gt shape: ", pose.shape)
    # print(motion['02_01'].keys())
    # file_path = "predict_3d_pos/gt"
    # write_standard_bvh(file_path, pose)