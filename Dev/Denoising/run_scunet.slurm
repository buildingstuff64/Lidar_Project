#!/bin/bash
#SBATCH --job-name=SCUNet
#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

nvidia-smi

conda init
source ~/.bashrc

conda activate SCUNet

cd ~/SCUNet

python main_test_scunet_real_application.py --model_name scunet_color_real_psnr --testset_name TESTSET_DIR
