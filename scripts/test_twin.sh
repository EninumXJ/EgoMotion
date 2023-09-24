source /home/litianyi/anaconda3/bin/activate ego
CUDA_VISIBLE_DEVICES=2,3 python tests/test.py --gpus 0,1  \
                                         --batch_size 1  \
                                         --val_size 100  \
                                         --name twin_test06  \
                                         --resume /home/litianyi/workspace/EgoMotion/checkpoints/twin_training06/rgb_checkpoint.pth  \
                                         --clip_length 10  \
                                         --log_dir /home/litianyi/workspace/EgoMotion/model/features  \
                                        #  --use_feature  \
                                        #  --feature_path /home/litianyi/workspace/EgoMotion/logs/tensorboard/twin_test01/feature_record.csv  \