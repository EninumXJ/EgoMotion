# CUDA_VISIBLE_DEVICES=1,2  python finetune.py  --config config/MB_twin_network_finetune.yaml  \
#                           -pb /home/litianyi/workspace/EgoMotion/MotionBERT/checkpoint/pretrain/latest_epoch.bin  \
#                           -pt /home/litianyi/workspace/EgoMotion/checkpoints/twin_training06/rgb_checkpoint.pth  \
#                           -r /home/litianyi/workspace/EgoMotion/checkpoints/exp03/latest_epoch.pth  \
#                           --use_wandb  \

CUDA_VISIBLE_DEVICES=1,2  python train_on_slam.py  --config config/DST_slam_train.yaml  \
                                                    --use_wandb  \
                                                    -c  /data/newhome/litianyi/model/EgoMotion/checkpoints/  \
                                                    --resume  /data/newhome/litianyi/model/EgoMotion/checkpoints/exp04/best_epoch.pth  \
                                                    