# CUDA_VISIBLE_DEVICES=1,2  python finetune.py  --config config/MB_twin_network_finetune.yaml  \
#                           -pb /home/litianyi/workspace/EgoMotion/MotionBERT/checkpoint/pretrain/latest_epoch.bin  \
#                           -pt /home/litianyi/workspace/EgoMotion/checkpoints/twin_training06/rgb_checkpoint.pth  \
#                           -r /home/litianyi/workspace/EgoMotion/checkpoints/exp03/latest_epoch.pth  \
#                           --use_wandb  \

CUDA_VISIBLE_DEVICES=1,2  python train_on_kinpoly.py   --log_dir /data/newhome/litianyi/logs/EgoMotion/  \
                                                       --config config/slam_train_kinpoly.yaml  \
                                                       --projector rnn  \
                                                       --use_wandb  \
                                                    #    --resume /data/newhome/litianyi/logs/EgoMotion/exp13/checkpoints/best_epoch.pth  \
                                                    