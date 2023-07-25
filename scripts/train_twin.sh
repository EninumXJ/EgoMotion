source /home/litianyi/anaconda3/bin/activate ego
CUDA_VISIBLE_DEVICES=2,3 python train.py --gpus 0,1  \
                                         --batch_size 8  \
                                         --n_epochs 30  \
                                         --train_size 5000  \
                                         --name twin_training06  \
                                         --L1 100  \
                                         --L2 5  \
                                         --epsilon 15  \
                                         --use_wandb  \