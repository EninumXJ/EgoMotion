source /home/litianyi/anaconda3/bin/activate ego
CUDA_VISIBLE_DEVICES=3 python evaluate.py  --config config/slam_train_kinpoly.yaml  \
                                           --resume /data/newhome/litianyi/model/KinPoly/checkpoints/exp02/best_epoch.pth  \
                                           --vis_path vis/vis_results/kinpoly02/demo03