import os
import torch
import torch.utils.data
import yaml
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
import joblib
import pytorch3d.transforms as transforms 
import sys
sys.path.append('..')

SMPLH_PATH = "/home/litianyi/workspace/egoego_release/smpl_models/smplh_amass/"

def run_smpl_model(root_trans, aa_rot_rep, betas, gender, bm_dict):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 16
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female']
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 
    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = pred_body.f 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 

def get_smpl_parents():
    bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 
    parents = ori_kintree_table[0, :22] # 22 
    parents[0] = -1 # Assign -1 for the root joint's parent idx.

    return parents

class KinPolyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, config_path, image_tmpl="{:05d}.jpg",
                 transform=None, clip_length=10, mode='train'):
        self.dataset_path = dataset_path
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(),Loader=yaml.FullLoader)
        if mode == 'train':
            self.action_list = config['train']
        elif mode == 'test':
            self.action_list = config['test']
        self.clip_length = clip_length
        mocap_path = os.path.join(self.dataset_path, 'mocap_annotations.p')
        self.mocap_data = joblib.load(mocap_path)
        self.transform = transform
        self.image_tmpl = image_tmpl
        self.preprocess(config)

    def preprocess(self, config):
        # length_list: 建立了索引到mocap数据的映射
        # pair_list: 动作-视频对
        self.action_index_range = []
        self.action_video_num = {}
        num_frames = 0; num = 0
        for action in self.action_list:
            # action: '02_01_walk'
            if config["video_mocap_sync"][action] == None:
                continue
            # self.action_video_num[action] = {}
            # index: '1', '2', ...
            mocap_frames = int(config["video_mocap_sync"][action][2] - config["video_mocap_sync"][action][1])
            # print("{0} mocap frames: {1}".format(action, mocap_frames))
            # mocap_frames_2 = int(config["mocap_frames"][action])
            # print("mocap_frames: ", mocap_frames)
            # print("mocap_frames_2: ", mocap_frames_2)
            useful_frames = mocap_frames - 2*self.clip_length  # 去掉首尾2个clip
            mocap_video_gap = int(config["video_mocap_sync"][action][0])
            mocap_start_frame = int(config["video_mocap_sync"][action][1])
            mocap_end_frame = int(config["video_mocap_sync"][action][2])
            self.action_video_num[num] = (action, mocap_video_gap, mocap_start_frame, mocap_end_frame)
            self.action_index_range.append(range(num_frames, num_frames+useful_frames))
            num += 1
            num_frames += useful_frames

    def _load_image(self, directory, index):
        return self.transform(Image.open(os.path.join(directory, self.image_tmpl.format(index))).convert('RGB'))

    def _load_image_clip(self, directory, index):
        images = []
        for i in range(index, index+self.clip_length):
            images.append(self._load_image(directory, i))
        clip = torch.stack(images, 0)
        return clip

    def _load_mocap(self, action, index):
        ### kinpoly pose: 23 joints + 1 root
        # print("index: ", index)
        # print("mocap frames: ", self.mocap_data[action]['qpos'].shape)
        joint_rot = self.mocap_data[action]['qpos'][index:index+self.clip_length, 7:]
        # print("joint_rot shape: ", joint_rot.shape)
        root = torch.from_numpy(self.mocap_data[action]['qpos'][index:index+self.clip_length, :7])
        print("root shape: ", root.shape)
        joint_rot = torch.from_numpy(joint_rot.reshape(self.clip_length, -1, 3))   # TxJx3
        # print("joint_rot shape: ", joint_rot.shape)
        # euler angle: (Z Y X) -> (X Y Z)
        # joint_rot = joint_rot[:,:,[2,1,0]]
        lrot_quat_status = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(joint_rot, convention='ZYX'))   ## TxJx4
        # print("lrot_quat_status shape: ", lrot_quat_status.shape)
        lrot_quat = transforms.quaternion_multiply(transforms.quaternion_invert(lrot_quat_status[:-1, ...]),
                                                   lrot_quat_status[1:, ...])
        # print("lrot_quat shape: ", lrot_quat.shape)
        ## lrot_quat表征该时刻的旋转相对于上一时刻的旋转量：以四元数表示
        
        zero = torch.zeros_like(lrot_quat[0:1,...])
        lrot_gt = torch.cat([zero, lrot_quat], dim=0)

        initial_rot = joint_rot

        return lrot_gt, root, initial_rot, joint_rot 

    def sample(self, input, ratio=0.5, mode='add'):
        new_num_frames = int(ratio * self.clip_length)
        downsamp_inds = np.linspace(0, self.clip_length, num=new_num_frames, endpoint=False, dtype=int)

        if mode == 'add':
            input_sample = input[downsamp_inds]
            downsamp_inds_odd = np.linspace(1, self.clip_length+1, num=new_num_frames, endpoint=False, dtype=int)
            input_sample_2 = input[downsamp_inds_odd]
            input_sample[1:] = transforms.quaternion_multiply(input_sample_2[:-1], input_sample[1:])
        elif mode == 'sample':
            input_sample = input[downsamp_inds]
        return input_sample

    def __len__(self):
        return self.action_index_range[-1][-1]
    
    def __getitem__(self, index):
        ind_bool = [index in i for i in self.action_index_range]
        # ind表示该index属于第ind个动作视频对
        ind = ind_bool.index(True)
        
        action, mocap_video_gap, mocap_start_frame, mocap_end_frame = self.action_video_num[ind]
        _, name = action.split('-', 1)
        video_path = os.path.join(self.dataset_path, 'images', name)
        index_in_video = index - self.action_index_range[ind][0] + mocap_start_frame + mocap_video_gap + self.clip_length
        index_in_mocap = index - self.action_index_range[ind][0] + self.clip_length  ###这里不需要加上start_frame，因为.p文件中已经将相应的mocap片段裁剪好了
        lrot_gt, root, initial_rot, joint_rot = self._load_mocap(action, index_in_mocap)
        img_clip = self._load_image_clip(video_path, index_in_video)
        # img_clip = self.sample(img_clip, mode='sample')
        # initial_rot = self.sample(initial_rot, mode='sample')
        # lrot_gt = self.sample(lrot_gt, mode='add')
        return lrot_gt, img_clip, root, initial_rot, joint_rot
    
    def get_rest_pose_joints(self):
        zero_root_trans = torch.zeros(1, 1, 3).cuda().float()
        zero_rot_aa_rep = torch.zeros(1, 1, 22, 3).cuda().float()
        bs = 1 
        betas = torch.zeros(1, 16).cuda().float()
        gender = ["male"] * bs 

        rest_human_jnts, _, _ = \
        run_smpl_model(zero_root_trans, zero_rot_aa_rep, betas, gender, self.bm_dict)
        # 1 X 1 X J X 3 

        parents = get_smpl_parents()
        parents[0] = 0 # Make root joint's parent itself so that after deduction, the root offsets are 0
        rest_human_offsets = rest_human_jnts.squeeze(0) - rest_human_jnts.squeeze(0)[:, parents, :]

        return rest_human_offsets # 1 X J X 3 

    def fk_smpl(self, root_trans, lrot_aa):
        # root_trans: N X 3 
        # lrot_aa: N X J X 3 

        # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
        # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
        
        parents = get_smpl_parents() 

        lrot_mat = transforms.axis_angle_to_matrix(lrot_aa) # N X J X 3 X 3 

        lrot = transforms.matrix_to_quaternion(lrot_mat)

        # Generate global joint position 
        lpos = self.rest_human_offsets.repeat(lrot_mat.shape[0], 1, 1) # T' X 22 X 3 

        gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
        for i in range(1, len(parents)):
            gp.append(
                transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
            )
            gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

        global_rot = torch.cat(gr, dim=-2) # T X 22 X 4 
        global_jpos = torch.cat(gp, dim=-2) # T X 22 X 3 

        global_jpos += root_trans[:, None, :] # T X 22 X 3

        return global_rot, global_jpos 


if __name__=='__main__':
    config_path = '/data/newhome/litianyi/dataset/kin_poly/MoCap_dataset/mocap_meta.yml'
    dataset_path = '/data/newhome/litianyi/dataset/kin_poly/MoCap_dataset'
    import torchvision
    img_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224)),
                    torchvision.transforms.ToTensor()])
    kinpolydata = KinPolyDataset(dataset_path=dataset_path, config_path=config_path, clip_length=16, transform=img_transforms)
    index = 177
    lrot_gt, img_clip, root, initial_rot = kinpolydata[index]
    print(lrot_gt.shape)
    print(img_clip.shape)
    print(root.shape)
    print(initial_rot.shape)

    #### small experiment
    import math
    axis_angle_1 = torch.tensor([[30./180*math.pi, 60/180*math.pi, 0]])   ## ZYX
    axis_angle_2 = torch.tensor([[60./180*math.pi, 90/180*math.pi, 30./180*math.pi]])   ## ZYX
    axis_angle = torch.cat([axis_angle_1, axis_angle_2], dim=0)
    print(axis_angle.shape)
    matrix = transforms.euler_angles_to_matrix(axis_angle, convention='ZYX')
    print("matrix: ", matrix)
    lrot_quat_status = transforms.matrix_to_quaternion(matrix)   ## TxJx4
    print("lrot_quat_status: ", lrot_quat_status)
    # print("lrot_quat_status shape: ", lrot_quat_status.shape)
    lrot_quat = transforms.quaternion_multiply(transforms.quaternion_invert(lrot_quat_status[:-1, ...]),
                                               lrot_quat_status[1:, ...])
    print("lrot_quat_1: ", lrot_quat)

    matrix_rot = torch.matmul(torch.linalg.inv(matrix[:-1,...]), matrix[1:,...])
    lrot_quat_2 = transforms.matrix_to_quaternion(matrix_rot)
    print("lrot_quat_2: ", lrot_quat_2)
    # from scipy.spatial.transform import Rotation as R
    # matrix_1 = R.from_euler('ZYX', [30./180*math.pi, 60/180*math.pi, 0], degrees=False).as_matrix()
    # print("matrix_1: ", matrix_1)
    # matrix_2 = R.from_euler('ZYX', [60./180*math.pi, 90/180*math.pi, 30./180*math.pi], degrees=False).as_matrix()
    # print("matrix_2: ", matrix_2)
    # quat_1 = R.from_matrix(matrix_1).as_quat()
    # print("quat_1: ", quat_1)
    # quat_2 = R.from_matrix(matrix_2).as_quat()
    # print("quat_2: ", quat_2)
    # matrix = np.matmul(np.linalg.inv(matrix_1), matrix_2)
    # quat = R.from_matrix(matrix).as_quat()
    # print(quat)