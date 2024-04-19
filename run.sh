#!/bin/bash
#SBATCH -J opt-ode
#SBATCH -o opt-ode.out
#SBATCH --cpus-per-task=12
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -N 1

cd /bicmr/home/zlxie/code
tensorboard --logdir=train_log --samples_per_plugin scalars=999999999
# python train.py --problem lpp --dataset a5a --num_epoch 80 --pen_coeff 0.5
# python train.py --problem lpp --dataset separable --num_epoch 15 --pen_coeff 0.5 --eps 1e-4
# python train.py --problem lpp --dataset covtype --num_epoch 50 --pen_coeff 0.5 --batch_size 10240
python train.py --problem logistic --dataset covtype --num_epoch 50 --pen_coeff 0.5 --batch_size 10240
# python train.py --problem lpp --dataset w3a --num_epoch 100 --pen_coeff 0.5