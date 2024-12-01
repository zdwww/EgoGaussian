#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --output=sbatch_log/%j.out
#SBATCH --error=sbatch_log/%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --constraint='titan_xp'

conda activate ego-3dgs
DATA_DIR=${HOME_DIR}/EgoGaussian-data
OUT_DIR=${HOME_DIR}/EgoGaussian-output
export PYTHONPATH=$PYTHONPATH:/home/daizhang/EgoGaussian

EK_NAMES=("P03_03" "P17_01" "P18_06" "P32_01")
HOI_NAMES=("Video1" "Video2" "Video3" "Video4" "Video5")

DATA_TYPE=EK # or HOI
DATA_NAME=P03_03
RUN_NAME=full
python train.py \
    --source_path ${DATA_DIR}/${DATA_TYPE}/${DATA_NAME} \
    --out_root ${OUT_DIR} \
    --data_type ${DATA_TYPE} \
    --video ${DATA_NAME} \
    --run_name ${RUN_NAME} \