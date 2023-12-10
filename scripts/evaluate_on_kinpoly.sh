source /home/litianyi/anaconda3/bin/activate ego
# CUDA_VISIBLE_DEVICES=3 python evaluate.py  --config config/slam_train_kinpoly.yaml  \
#                                            --resume /data/newhome/litianyi/model/KinPoly/checkpoints/exp02/best_epoch.pth  \
#                                            --vis_path vis/vis_results/kinpoly02/demo03

CUDA_VISIBLE_DEVICES=3 python tests/test_on_local_stage.py  --log_dir /data/newhome/litianyi/logs/EgoMotion/  \
                                                            --config config/test_on_kinpoly.yaml  \
                                                            --local_resume /data/newhome/litianyi/logs/EgoMotion/exp12/checkpoints/best_epoch.pth  \
                                                            --projector rnn  \
                                                            --vis_path vis_results/kinpoly12/demo01