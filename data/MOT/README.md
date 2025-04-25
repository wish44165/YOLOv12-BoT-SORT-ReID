## MOT (Multiple Object Tracking)

This section provides datasets related to the MOT task (Track 3 of the [4th Anti-UAV Workshop](https://anti-uav.github.io/)), including the processed `Strong Baseline` [datasets](https://doi.org/10.5281/zenodo.15203123) presented in Table 2 of the [paper](https://arxiv.org/pdf/2503.17237) and the `Officially Released` datasets from [Track 3](https://codalab.lisn.upsaclay.fr/competitions/21806#participate), available via the links below.


### (1) Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15203123.svg)](https://doi.org/10.5281/zenodo.15203123)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/wish44165/StrongBaseline_YOLOv12-BoT-SORT-ReID)

You can download `AntiUAV_train_val.zip` from the `MOT/` folder [here](https://doi.org/10.5281/zenodo.15203123). After extracting, rename the folder to `datasets/` and follow the folder structure below.

- AntiUAV_train_val.zip

```
datasets/
├── train/*.jpg, *.txt
└── val/*.jpg, *.txt
```


### (0) Official Dataset Release for the 4th Anti-UAV Challenge

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15103888.svg)](https://doi.org/10.5281/zenodo.15103888)

You can download the officially released datasets from the following links:

- Train
    - [Google Drive](https://drive.google.com/drive/folders/1JvGdAJjGzjOIGMG82Qiz5YJKzjy8VOd-?usp=drive_link) | [Baidu](https://pan.baidu.com/s/19iVwI1MW9OdXyPIc0xBSjQ?from=init&pwd=CVPR)
- Test
    - [Google Drive](https://drive.google.com/drive/folders/1cfF00w_3ewUMELSSnmaYOKLTZoIWlxbF?usp=sharing) | [Baidu](https://pan.baidu.com/s/1rhB24tksTw1JW6ZltOSvOg?pwd=CVPR)

```
.
└── MultiUAV_Test/
```