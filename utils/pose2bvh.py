import os
import sys
sys.path.append("utils/")
from cmu_skeleton import *

### https://github.com/HW140701/VideoTo3dPoseAndBvh/blob/master/videopose.py
def write_standard_bvh(outbvhfilepath, prediction3dpoint):
    '''
    :param outbvhfilepath: 输出bvh动作文件路径
    :param prediction3dpoint: 预测的三维关节点
    :return:
    '''

    # 将预测的点放大100倍
    for frame in prediction3dpoint:
        for point3d in frame:
            point3d[0] *= 3
            point3d[1] *= 3
            point3d[2] *= 3

            # 交换Y和Z的坐标
            #X = point3d[0]
            #Y = point3d[1]
            #Z = point3d[2]

            #point3d[0] = -X
            #point3d[1] = Z
            #point3d[2] = Y

    dir_name = os.path.dirname(outbvhfilepath)
    basename = os.path.basename(outbvhfilepath)
    video_name = basename[:basename.rfind('.')]
    bvhfileDirectory = os.path.join(dir_name,video_name,"bvh")
    if not os.path.exists(bvhfileDirectory):
        os.makedirs(bvhfileDirectory)
    bvhfileName = os.path.join(dir_name,video_name,"bvh","{}.bvh".format(video_name))
    cmu_skeleton = CMUSkeleton()
    cmu_skeleton.poses2bvh(prediction3dpoint, output_file=bvhfileName)