<p align="center">
  <h2 align="center">EgoGaussian: Dynamic Scene Understanding from Egocentric Video with 3D Gaussian Splatting</h2>
  <h5 align="center">International Conference on 3D Vision (3DV) 2025</h5>
</p>

<div align="center"> 

[Project Page](https://zdwww.github.io/egogs.github.io/) | [Paper](https://arxiv.org/abs/2406.19811) | [Video](https://www.youtube.com/watch?v=nsZrmM7CJB0) | [Data](https://drive.google.com/file/d/1VCC71f7YYeCahQlSNpJ0BsR1995W6jDI/view?usp=sharing)

  <img src="assets/egogs.gif">
</div>

## Updates
- <b>[06/20/2025] Data Correction</b>: The ID in `submission/HOI4D/Video1/id.txt` is incorrect. It should be `ZY20210800001/H1/C8/N11/S321/s03/T2`. Thx to [gaperezsa](https://github.com/gaperezsa) for identifying and reporting this issue.
- <b>[12/01/2024]</b>  Initial code release

## 📝 TODO List
- \[x\] Release code of EgoGaussian 
- \[x\] Release 3DGS-ready egocentric data we processed from [EPIC-KITCHENS](https://epic-kitchens.github.io/2024), [HOI4D](https://hoi4d.github.io), and [EPIC Fields](https://epic-kitchens.github.io/epic-fields/). Please also consider citing their great works if you use this subset 🤗
- \[ \] Upload pre-trained checkpoints for quick evaluation and visualization
- \[ \] EgoGaussian viewer
- \[ \] Pipeline optimization
- \[x\] Tutorial for running EgoGaussian on customized data

## 🛠️ Setup
The setup should be very similar to the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) except we used a modified version of [differential gaussian rasterization](https://github.com/ashawkey/diff-gaussian-rasterization/tree/8829d14f814fccdaf840b7b0f3021a616583c0a1) with support of depth & alpha rendering. We will release the `requirements.txt` later.

## 🎥 Dataset

[Data](https://drive.google.com/file/d/1VCC71f7YYeCahQlSNpJ0BsR1995W6jDI/view?usp=sharing) used for EgoGaussian is structured as follows:

```bash
Submission/ 
├── HOI4D/ 
│   └──Video1/ 
│      ├──images/ # video frames (300 for HOI4D)
│      │  ├──00000.png 
│      │  └──xxxxx.png
│      ├──sparse/ # camera intrinsics/extrinsics in COLMAP format
│      ├──obj_masks/ # mask where object pixels are white and the rest of the image is black; may be missing for some frames.
│      ├──hand_masks/ # mask where hand pixels are white and rest of the image is black; must be ene mask per video frame
│      ├──split/
│      │  ├──training_frames.txt # indices for training frames: 5 digits for HOI4D, 10 for EK
│      │  ├──static_eval_frames.txt # frame indices in static clip used for evaluation
│      │  ├──dynamic_eval_frames.txt 
│      │  └──phase_frame_index.txt # record the alternating between static and dynamic clip, e.g. frame 0 to 55 are static, frame 56 to 139 are dynamic
│      └──id.txt # video path in HOI4D dataset
└── EK /
    └──P03_03/ # index in EPIC-KITCHENS
       ├──images/ # video frames (variable numbe)
       │  ├──frame_0000003880.png # same index as original EK dataset
       │  └──frame...
       └──frames.txt # frame range
Webpage/ # same structure
```
where `Submission` folder contains 5 videos from HOI4D and 4 videos from EPIC-KITCHENS, which are used to generate the results in Table 1 and Figure 3 of the paper, `Webpage` folder contains 2 additional videos from HOI4D used as demonstration videos on the project webpage.

Note ‼: On page 6 of our paper, we stated ‘we randomly select 4 videos (from HOI4D)’, which is incorrect. We actually used all 5 videos listed in our dataset to generate the results. Also, the `id.txt` in `submission/HOI4D/Video1` is incorrect and should be `ZY20210800001/H1/C8/N11/S321/s03/T2`

To run our pipeline on custom data, follow these steps and ensure the preprocessed data matches our format:
1. Run [EgoHOS](https://github.com/owenzlz/EgoHOS) to identify hand segmentation
2. Run a pipeline similar to [EPIC Fields](https://github.com/epic-kitchens/epic-fields-code)
 to obtain camera poses. While excluding the hand segmentation from the previous step is recommended, it is not strictly necessary.
3. Select the object several frames right before the interaction and run [Track Anything](https://github.com/gaomingqi/Track-Anything) to segment the interacted objects

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