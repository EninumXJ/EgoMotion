CUDA_VISIBLE_DEVICES=1,2  python finetune.py  --config config/MB_twin_network_finetune.yaml  \
                          -pb /home/litianyi/workspace/EgoMotion/MotionBERT/checkpoint/pretrain/latest_epoch.bin  \
                          -pt /home/litianyi/workspace/EgoMotion/checkpoints/twin_training06/rgb_checkpoint.pth  \
                          --use_wandb