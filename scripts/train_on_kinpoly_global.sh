CUDA_VISIBLE_DEVICES=3  python train_on_kinpoly_global.py  --config config/slam_train_kinpoly_global.yaml  \
                                                            -c  /data/newhome/litianyi/model/KinPoly/checkpoints/  \
                                                            --use_wandb  \