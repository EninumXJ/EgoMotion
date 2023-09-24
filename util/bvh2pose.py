from util.bvh2np import Bvh
import sys
from glob import glob
import os
import joblib
import numpy as np

### 21 Joints
Bone_name_list = ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RHipJoint', 'RightUpLeg', 
                  'RightLeg', 'RightFoot', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 
                  'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']

Bone_Addr = [0, 1, 2, 3, 4, 7, 8, 9, 10, 14, 15, 16, 18, 20, 21, 22, 23, 29, 30, 31, 32]
Bone_Addr_17joints = [0, 2, 3, 4, 6, 7, 8, 10, 11, 9, 12, 14, 15, 16, 18, 19, 20]
# hips, Lupleg, Lleg, Lfoot, Rupleg, Rleg, Rfoot, Spine1, Neck, chin, head, Larm, Lforearm, Lhand, Rarm, Rforearm, Rhand
Switch_Position = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def convert_bvh(mocap_path, output_path, fps=30):
    ratio = float(fps) / 120.
    bvh_files = glob(os.path.join(mocap_path, '*'))
    print(bvh_files)
    mocap_pose = {}
    for bvh_file in bvh_files:
        print("bvh_path: ", bvh_file)
        anim = Bvh()
        anim.parse_file(bvh_file)
        print("total frames: ", anim.frames)
        trans = []
        rotation = []
        (path, bvh_filename) = os.path.split(bvh_file)
        bvh_filename = bvh_filename[:-4]
        print("bvh_filename: ", bvh_filename)
        mocap_pose[bvh_filename] = {}
        ### downsample
        num_frames = anim.frames
        new_num_frames = int(ratio*num_frames)
        print("new frames: ", new_num_frames)
        downsamp_inds = np.linspace(0, num_frames-1, num=new_num_frames, dtype=int)
        print(downsamp_inds)
        for frame in downsamp_inds:
            positions, rotations = anim.frame_pose(frame)
            trans.append(np.array(positions))
            rotation.append(np.array(rotations))
            # trans.append(np.array([np.array(positions[i]) for i in Bone_Addr]))
            # rotation.append(np.array([np.array(rotations[i]) for i in Bone_Addr]))
        trans_np = np.array(trans)
        rotation_np = np.array(rotation)
        print("trans_np shape: ", trans_np.shape)
        print("rotation_np shape: ", rotation_np.shape)
        mocap_pose[bvh_filename]['trans'] = trans_np
        mocap_pose[bvh_filename]['rotation'] = rotation_np 
    output_filename = os.path.join(output_path, 'egomotion_pose_30fps_38joints.p')
    joblib.dump(mocap_pose, open(output_filename, "wb"))
    

if __name__ == '__main__':
    mocap_root_path = '../data/EgoMotion/bvh/'
    output_path = ''
    convert_bvh(mocap_root_path, output_path)
    # bvh_file = os.path.join(mocap_root_path, '02_01.bvh')
    # anim = Bvh()
    # anim.parse_file(bvh_file)
    # print(anim.joint_names)
    # print(len(anim.frame_pose(10)[0]))
