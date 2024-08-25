#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=m2a_bt
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2anomaly_v2/train_bt.log
#SBATCH --error=/home/nberardo/mask2anomaly_v2/train_bt.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

#convert pth weights to pkl
python tools/convert-torchvision-to-d2.py /home/nberardo/barlowtwins/checkpoint/resnet50.pth /home/nberardo/mask2anomaly_v2/backbone_weights/bt_resnet50.pkl

#CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file configs/cityscapes/semantic-segmentation/anomaly_train.yaml