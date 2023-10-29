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

def output2quat(lrot_quat, joint_rot_initial):
        clip_len = lrot_quat.shape[1]
        lrot_quat_initial = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(joint_rot_initial[:,0:1,...], convention='ZYX')) ### (B,T,J,3)->（B,T,J,4）
        lrot_quat_status = []
        lrot_quat_status.append(lrot_quat_initial)
        for i in range(0, clip_len):
            curr_quat = lrot_quat_status[-1]
            lrot_quat_status.append(transforms.quaternion_multiply(curr_quat, lrot_quat[:, i:i+1,...]))
            # print(lrot_quat_status[-1].shape)
        return torch.stack(lrot_quat_status, dim=1).squeeze(2)

def output2matrix(lrot_rot6d, joint_rot_initial):
        clip_len = lrot_rot6d.shape[1]
        joint_matrix = transforms.euler_angles_to_matrix(joint_rot_initial[:,0:1,...], convention='ZYX') ### (B,T,J,3)->（B,T,J,3,3）
        lrot_matrix = []
        lrot_matrix.append(joint_matrix)
        for i in range(0, clip_len):
            curr_matrix= lrot_matrix[-1]
            lrot_matrix.append(torch.matmul(curr_matrix, transforms.rotation_6d_to_matrix(lrot_rot6d[:, i:i+1,...])))
            # print(lrot_quat_status[-1].shape)
        return torch.stack(lrot_matrix, dim=1).squeeze(2)

def collate_function(data):
    output = list(zip(*data))
    lrot_gt, img_clip, root, joint_rot, name_index = output[0], output[1], output[2], output[3], output[4]
    lrot_gt = torch.stack(lrot_gt, 0)
    img_clip = torch.stack(img_clip, 0)
    root = torch.stack(root, 0)
    joint_rot = torch.stack(joint_rot, 0)
    video_index = list(name_index)
    return (lrot_gt, img_clip, root, joint_rot, video_index)


class KinPolyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, config_path, image_tmpl="{:05d}.jpg",
                 transform=None, clip_length=10, mode='train', use_slam=False, rot='rot6d',
                 coordinate='local'):
        self.dataset_path = dataset_path
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(),Loader=yaml.FullLoader)
        if mode == 'train':
            self.action_list = config['train']
        elif mode == 'test':
            self.action_list = config['test']
        self.mode = mode
        self.clip_length = clip_length
        mocap_path = os.path.join(self.dataset_path, 'mocap_annotations.p')
        self.mocap_data = joblib.load(mocap_path)
        self.transform = transform
        self.image_tmpl = image_tmpl
        self.rot = rot
        self.coordinate = coordinate
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
            _, name = action.split('-', 1)
            img_path = os.path.join(self.dataset_path, 'images', name)
            files = os.listdir(img_path)   # 读入文件夹
            video_end_frame = len(files)       # 统 计文件夹中的文件个数
            # index: '1', '2', ...
            total_frames = min(int(config["video_mocap_sync"][action][2]), video_end_frame) - int(config["video_mocap_sync"][action][1])
            # print("{0} mocap frames: {1}".format(action, mocap_frames))
            # mocap_frames_2 = int(config["mocap_frames"][action])
            # print("mocap_frames: ", mocap_frames)
            # print("mocap_frames_2: ", mocap_frames_2)
            useful_frames = total_frames - 2*self.clip_length  # 去掉首尾2个clip
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
        # print("mocap frames: ", self.mocap_data[action]['qpos'].shape)
        joint_rot = self.mocap_data[action]['qpos'][index:index+self.clip_length, 7:]
        # print("joint_rot shape: ", joint_rot.shape)
        root = torch.from_numpy(self.mocap_data[action]['qpos'][index:index+self.clip_length, :7]).float()
        # print("root shape: ", root.shape)
        joint_rot = torch.from_numpy(joint_rot.reshape(self.clip_length, -1, 3)).float()   # TxJx3
        # print("joint_rot: ", joint_rot) 
        # euler angle: (Z Y X) -> (X Y Z)
        # joint_rot = joint_rot[:,:,[2,1,0]]
        if self.rot == 'quat4d':
            lrot_quat_status = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(joint_rot, convention='ZYX'))   ## T x J x 4
            
            lrot_quat = transforms.quaternion_multiply(transforms.quaternion_invert(lrot_quat_status[:-1, ...]),
                                                    lrot_quat_status[1:, ...])
            # print("lrot_quat shape: ", lrot_quat.shape)
            ## lrot_quat表征该时刻的旋转相对于上一时刻的旋转量：以四元数表示
            
            zero = torch.zeros_like(lrot_quat[0:1,...])
            lrot_gt = torch.cat([zero, lrot_quat], dim=0)
        elif self.rot == 'rot6d':
            lrot_matrix = transforms.euler_angles_to_matrix(joint_rot, convention='ZYX')   # (T) X J x 3 X 3
            ##### copy from: https://github.com/lijiaman/egoego_release/blob/4dae8cb67b453f0a0042809bf37aafad5604615b/utils/data_utils/process_kinpoly_qpos2smpl.py#L498C8-L498C88
            lrot_matrix_diff = torch.matmul(torch.linalg.inv(lrot_matrix[:-1, ...]), lrot_matrix[1:, ...])  # (T-1) X J X 3 X 3
            # 3 X 3 matrix -> 6d rotation
            lrot_6d = transforms.matrix_to_rotation_6d(lrot_matrix_diff)
            zero = torch.zeros_like(lrot_6d[0:1, ...])  
            lrot_gt = torch.cat([zero, lrot_6d], dim=0)   # T X J X 6
        return lrot_gt, root, joint_rot

    def _load_root_traj(self, action, index):
        root_traj = torch.from_numpy(self.mocap_data[action]['qpos'][index:index+self.clip_length, :7]).float()   # T X 7
        root_trans = root_traj[..., 0:3]    # T X 3
        root_quat = root_traj[..., 3:]      # T X 4
        root_trans_diff = root_trans[1:, ...] - root_trans[:-1, ...]  # (T-1) X 3
        if self.rot == 'rot6d':
            root_quat_diff = transforms.quaternion_multiply(transforms.quaternion_invert(root_quat[:-1, ...]), root_quat[1:, ...])
            root_rot6d_diff = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(root_quat_diff))
            root_rot_diff = root_rot6d_diff   # (T-1) X 6
        elif self.rot == 'quat4d':
            root_quat_diff = transforms.quaternion_multiply(transforms.quaternion_invert(root_quat[:-1, ...]), root_quat[1:, ...])
            root_rot_diff = root_quat_diff    # (T-1) X 4
        # initial_traj = root_traj[0]
        root_trans_zeros = torch.zeros_like(root_trans_diff[0:1])
        root_rot_zeros = torch.zeros_like(root_rot_diff[0:1])
        root_trans_diff = torch.cat([root_trans_zeros, root_trans_diff], dim=0)  # T X 3
        root_rot_diff = torch.cat([root_rot_zeros, root_rot_diff], dim=0)  # T X 4 or T X 6
        # print("root_trans_diff shape: ", root_trans_diff.shape)
        # print("root_rot_diff shape: ", root_rot_diff.shape)
        return root_trans_diff, root_rot_diff, root_traj

    def sample(self, input, ratio=0.5, mode='add'):
        new_num_frames = int(ratio * self.clip_length)
        downsamp_inds = np.linspace(0, self.clip_length, num=new_num_frames, endpoint=False, dtype=int)

        if mode == 'add':
            if input.shape[-1] == 3:   ### trans
                input_sample = input[downsamp_inds]
                downsamp_inds_odd = np.linspace(1, self.clip_length+1, num=new_num_frames, endpoint=False, dtype=int)
                input_sample_2 = input[downsamp_inds_odd]
                input_sample[1:] = input_sample_2[:-1] + input_sample[1:]
            else:
                input_sample = input[downsamp_inds]
                downsamp_inds_odd = np.linspace(1, self.clip_length+1, num=new_num_frames, endpoint=False, dtype=int)
                input_sample_2 = input[downsamp_inds_odd]
                if self.rot == 'quat4d':
                    input_sample[1:] = transforms.quaternion_multiply(input_sample_2[:-1], input_sample[1:])
                elif self.rot == 'rot6d':
                    input_sample_matrix = transforms.rotation_6d_to_matrix(input_sample)
                    input_sample_2_matrix = transforms.rotation_6d_to_matrix(input_sample_2)
                    input_sample[1:] = transforms.matrix_to_rotation_6d(torch.matmul(input_sample_2_matrix[:-1], input_sample_matrix[1:]))

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
        if self.coordinate == 'local':
            lrot_gt, root, joint_rot = self._load_mocap(action, index_in_mocap)
            img_clip = self._load_image_clip(video_path, index_in_video)
            img_clip = self.sample(img_clip, mode='sample')
            lrot_gt = self.sample(lrot_gt, mode='add')
            joint_rot = self.sample(joint_rot, mode='sample')
            root = self.sample(root, mode='sample')
            if self.mode == 'train':
                return lrot_gt, img_clip, root, joint_rot
            elif self.mode == 'test':
                return lrot_gt, img_clip, root, joint_rot, (name, index_in_video)
        elif self.coordinate == 'global':
            root_trans_diff, root_rot_diff, root_traj = self._load_root_traj(action, index_in_mocap)
            img_clip = self._load_image_clip(video_path, index_in_video)
            img_clip = self.sample(img_clip, mode='sample')
            root_traj = self.sample(root_traj, mode='sample')
            root_trans_diff = self.sample(root_trans_diff, mode='add')
            root_rot_diff = self.sample(root_rot_diff, mode='add')
            if self.mode == 'train':
                return img_clip, root_trans_diff[1:,...], root_rot_diff[1:,...], root_traj
            elif self.mode == 'test':
                return img_clip, root_trans_diff[1:,...], root_rot_diff[1:,...], root_traj, (name, index_in_video)
    
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
    kinpolydata = KinPolyDataset(dataset_path=dataset_path, config_path=config_path, clip_length=16, 
                                 transform=img_transforms, rot='rot6d', coordinate='global')
    index = 177
    img_clip, root_trans_diff, root_rot_diff, initial_rot = kinpolydata[index]
    print("img_clip: ", img_clip.shape)
    print("root_trans_diff: ", root_trans_diff.shape)
    print("root_rot_diff: ", root_rot_diff.shape)
    print("initial_rot shape: ", initial_rot.shape)
    
    # #### small experiment
    # import math
    # axis_angle_1 = torch.tensor([[30./180*math.pi, 60/180*math.pi, 0]])   ## ZYX
    # axis_angle_2 = torch.tensor([[60./180*math.pi, 90/180*math.pi, 30./180*math.pi]])   ## ZYX
    # axis_angle = torch.cat([axis_angle_1, axis_angle_2], dim=0)
    # print(axis_angle.shape)
    # matrix = transforms.euler_angles_to_matrix(axis_angle, convention='ZYX')
    # print("matrix: ", matrix)
    # lrot_quat_status = transforms.matrix_to_quaternion(matrix)   ## TxJx4
    # print("lrot_quat_status: ", lrot_quat_status)
    # # print("lrot_quat_status shape: ", lrot_quat_status.shape)
    # lrot_quat = transforms.quaternion_multiply(transforms.quaternion_invert(lrot_quat_status[:-1, ...]),
    #                                            lrot_quat_status[1:, ...])
    # print("lrot_quat_1: ", lrot_quat)

    # matrix_rot = torch.matmul(torch.linalg.inv(matrix[:-1,...]), matrix[1:,...])
    # lrot_quat_2 = transforms.matrix_to_quaternion(matrix_rot)
    # print("lrot_quat_2: ", lrot_quat_2)
    
    joint_quat_status = kinpolydata.output2quat(lrot_gt, initial_rot)
    print(joint_quat_status.shape)
    joint_euler_status = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(joint_quat_status), convention='ZYX')
    print("joint_euler_status: ", joint_euler_status) 