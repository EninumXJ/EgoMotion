CUDA_VISIBLE_DEVICES=3 python tests/tests_on_two_stage.py  --config config/slam_train_kinpoly.yaml  \
                                                            --local_resume /data/newhome/litianyi/model/KinPoly/checkpoints/exp04/best_epoch.pth  \
                                                            --global_resume /data/newhome/litianyi/model/KinPoly/checkpoints/global_exp01/best_epoch_global.pth  \
                                                            --vis_path vis/vis_results/kinpoly03/two-stage/demo01