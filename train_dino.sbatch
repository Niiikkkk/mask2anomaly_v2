#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=m2a_dino
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2anomaly_v2/train_dino.log
#SBATCH --error=/home/nberardo/mask2anomaly_v2/train_dino.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

#convert pth weights to pkl
python tools/convert-torchvision-to-d2.py /home/nberardo/mask2anomaly_v2/backbone_weights/dino_resnet50_800epochs.pth /home/nberardo/mask2anomaly_v2/backbone_weights/dino_resnet50_800epochs.pkl

#CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file configs/cityscapes/semantic-segmentation/anomaly_train.yaml