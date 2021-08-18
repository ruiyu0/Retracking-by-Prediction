# Towards Robust Human Trajectory Prediction in Raw Videos
The repository contains the code and data for "Towards Robust Human Trajectory Prediction in Raw Videos" IROS 2021.

## Dataset 

### SDD
Download the Stanford Drone Dataset (SDD) from the [official website](https://cvgl.stanford.edu/projects/uav_data/). The ~69G zip file includes the raw videos and annotations.
- **Our refined annotations of SDD**
We found many outlier bounding boxes in the original annotations via visualization. To better evaluate the tracking, we refined the annotations by removing the outliers. Our refined annotation files have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1_hMSdAr31l5XoZj3SmW_9QShd_p99yGW?usp=sharing).

### WILDTRACK
Download the WILDTRACK Seven-Camera HD Dataset from the [official website](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/).
- **Our extended annotations of WILDTRACK**
The original dataset annotated the first 400 frames of each of the seven videos at 2 fps. To better facilitate the behavior modeling, we manually annotated the first 900 frames (refined 1~400 and new 401~900) for each video with the [annotation tool](https://github.com/cvlab-epfl/multicam-gt). The extended annotation files have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1vVXNmbuOCx4qWyNVfTXPH_8b5xDP4YOj?usp=sharing), including the original WILDTRACK format and the [MOT format](https://motchallenge.net/instructions/). Our annotations can be used for larger-scale evaluation on WILDTRACK for single-view/multi-view pedestrian detection, tracking, and trajectory prediction.

## Installation
The project was developed based on Python 3.7.9 and PyTorch 1.1.0. The environment can be set up via the [environment.yml](environment.yml) file:
```
conda env create -f environment.yml
conda activate retracking
```

## Experiments
As an example, we show how to run the experiments on the SDD dataset.

1. **Train a prediction model**
    ```bash
    cd retracking-by-prediction
    python train.py
    ```

2. **Detection, Tracking, Prediction, and Re-tracking**
    ```bash
    sh run_test.sh
    ```

3. **Evaluation**
    We evaluate the tracking performance via [py-motmetrics](https://github.com/cheind/py-motmetrics). Note: we use L2 distance in meter instead of box IoU as the association metric.

## Citation
If you use the code or data in your research, please cite the paper:
```
@inproceedings{Yu2021-Retracking,
  author    = {Rui Yu and Zihan Zhou},
  title     = {Towards Robust Human Trajectory Prediction in Raw Videos},
  booktitle = {{IEEE/RSJ} International Conference on Intelligent Robots and Systems, {IROS}},
  year      = {2021}
}
```

## License
The project is released under the MIT License. The [SORT](https://github.com/abewley/sort) tracking code should follow its own license (GPL-3.0).
