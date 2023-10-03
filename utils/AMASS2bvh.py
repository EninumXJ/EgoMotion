import sys
sys.path.append('../AnimationTool/')
sys.path.append('..')
sys.path.append('../AnimationTool/anim')
sys.path.append('../AnimationTool/util')
from AnimationTool.anim import bvh
from AnimationTool.anim import amass
from anim.animation import Animation

from glob import glob
import os.path
from os.path import exists, isdir, basename, join, splitext

if __name__=='__main__':
    data_root = "/data/newhome/litianyi/dataset/AMASS"
    smplh_path = "/home/litianyi/workspace/egoego_release/smpl_models/smplh_amass/neutral/model.npz"
    output_root = "/data/newhome/litianyi/dataset/AMASS_bvh"

    sub_dataset_dir = [files for files in glob(data_root + "/*") if isdir(files)]
    complete_dataset = [os.path.split(files)[1] for files in glob(output_root + "/*") if isdir(files)]
    print("complete_dataset: ", complete_dataset)
    seq_dir_name = []
    for sub_dataset in sub_dataset_dir:
        seq_path = os.path.join(data_root, sub_dataset)
        (path, sub_name) = os.path.split(sub_dataset)
        if sub_name in complete_dataset:
            continue
        if sub_name == 'SOMA':
            continue
        print("sub_dataset: ", sub_name)
        seq_dir_name = [files for files in glob(seq_path + "/*") if isdir(files)]
        for seq_dir in seq_dir_name:
            (path, seq_name) = os.path.split(seq_dir)
            print("seq_name: ", seq_name)
            seq = [files for files in glob(seq_dir + "/*")]
            for sequence in seq:
                (path, sequence_name) = os.path.split(sequence)
                sequence_name = sequence_name[:-4]     # "xxxx.npz"->"xxx"
                print("Sequence_name: ", sequence_name)
                if sequence_name != 'shape' and sequence_name != 'male_stagei':
                    anim: Animation = amass.load(amass_motion_path=sequence, 
                                                smplh_path=smplh_path)
                    
                    bvh_path_base = '/data/newhome/litianyi/dataset/AMASS_bvh'
                    bvh_dir = os.path.join(bvh_path_base, sub_name, seq_name)
        
                    os.makedirs(bvh_dir, exist_ok=True)
                    bvh_path = os.path.join(bvh_dir, sequence_name + ".bvh")
                    bvh.save(filepath=bvh_path, anim=anim)