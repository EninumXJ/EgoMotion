import utils.cmu_skeleton as cmu
bvh_file_path = '/home/litianyi/workspace/EgoMotion/data/EgoMotion/bvh/02_01.bvh'
skeleton = cmu.CMUSkeleton()
skeleton.get_bone_length(bvh_file_path)