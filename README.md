<p align="center">
  <h2 align="center">EgoGaussian: Dynamic Scene Understanding from Egocentric Video with 3D Gaussian Splatting</h2>
  <h5 align="center">International Conference on 3D Vision (3DV) 2025</h5>
</p>

<div align="center"> 

[Project Page](https://zdwww.github.io/egogs.github.io/) | [Paper](https://arxiv.org/abs/2406.19811) | [Video](https://www.youtube.com/watch?v=nsZrmM7CJB0) | [Data](https://drive.google.com/file/d/1VCC71f7YYeCahQlSNpJ0BsR1995W6jDI/view?usp=sharing)

  <img src="assets/egogs.gif">
</div>

## Updates
- <b>[12/01/2024]</b>  Initial code release

## 📝 TODO List
- \[x\] Release code of EgoGaussian 
- \[x\] Release 3DGS-ready egocentric data we processed from [EPIC-KITCHENS](https://epic-kitchens.github.io/2024), [HOI4D](https://hoi4d.github.io), and [EPIC Fields](https://epic-kitchens.github.io/epic-fields/). Please also consider citing their great works if you use this subset 🤗
- \[ \] Upload pre-trained checkpoints for quick evaluation and visualization
- \[ \] EgoGaussian viewer
- \[ \] Pipeline optimization
- \[ \] Tutorial for running EgoGaussian on customized data

## 🛠️ Setup
The setup should be very similar to the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) except we used a modified version of [differential gaussian rasterization](https://github.com/ashawkey/diff-gaussian-rasterization/tree/8829d14f814fccdaf840b7b0f3021a616583c0a1) with support of depth & alpha rendering. We will release the `requirements.txt` later.

## Overview

The full EgoGaussian pipelie consists of 4 main stages corresponding to different scripts under `trainers`

1. Static object & background initialization
2. Coarse object pose estimation
3. Fine-tuning object pose & shape
4. Fine-tuning full dynamic scene

## Quick start

You can use the following script to run a full EgoGaussian pipeline from scratch on the provided data.
```shell
sbatch train.sh
```

## Reproducing the results

You can also skip the training and directly reproducing the results of Table 1 in our paper and videos on the webpage by running the following script with the checkpoints we provide.
```shell
DATA_TYPE=EK # or HOI
DATA_NAME=P03_03 # or Video0
RUN_NAME=full
python eval.py \
    --source_path ${DATA_DIR}/${DATA_TYPE}/${DATA_NAME} \
    --out_root ${OUT_DIR} \
    --data_type ${DATA_TYPE} \
    --video ${DATA_NAME} \
    --run_name ${RUN_NAME} \
```

## Acknowledgement
Our implementation is heavily based on the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). We thank the authors for their revolutionary work and open-source contributions. 

## Citation
If you find our paper useful, please cite us:
```bib
@misc{zhang2024egogaussiandynamicsceneunderstanding,
      title={EgoGaussian: Dynamic Scene Understanding from Egocentric Video with 3D Gaussian Splatting}, 
      author={Daiwei Zhang and Gengyan Li and Jiajie Li and Mickaël Bressieux and Otmar Hilliges and Marc Pollefeys and Luc Van Gool and Xi Wang},
      year={2024},
      eprint={2406.19811},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.19811}, 
}