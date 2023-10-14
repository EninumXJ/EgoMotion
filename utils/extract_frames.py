import os
from pathlib import Path


if __name__=='__main__':
    data_root = "/data/newhome/litianyi/dataset/kin_poly/MoCap_dataset/"
    mocap_video_path = os.path.join(data_root, 'MocapVideos')
    video_list = os.listdir(mocap_video_path)
    # print(video_list)
    for video in video_list:
        video_name = video[:-4]
        print(video_name)
        video_path = os.path.join(mocap_video_path, video)
        print("video path: ", video_path)
        img_path = os.path.join(data_root, 'images', video_name)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        # os.system("cd {}".format(video_path))
        os.system("ffmpeg -i {} -r 30 -q:v 2 -f image2 {}".format(video_path,
                                                                   img_path+'/'+'%05d.jpg'))
        