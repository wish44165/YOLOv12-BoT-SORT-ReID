## SOT (Single Object Tracking)

This section provides datasets related to the SOT task (Track 1 and Track 2 of the [4th Anti-UAV Workshop](https://anti-uav.github.io/)), including the processed `Strong Baseline` [datasets](https://zenodo.org/records/15203123) presented in Table 2 of the [paper](https://arxiv.org/pdf/2503.17237) and the `Officially Released` datasets from [Track 1](https://codalab.lisn.upsaclay.fr/competitions/21688#participate-get_data) and [Track 2](https://codalab.lisn.upsaclay.fr/competitions/21690#participate), available via the links below.


### (1) Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15203123.svg)](https://doi.org/10.5281/zenodo.15203123)

You can download either `AntiUAV_train_val.zip` or `AntiUAV_train_val_test.zip` from the `SOT/` folder [here](https://doi.org/10.5281/zenodo.15203123). After extracting, rename the folder to `datasets/` and follow the folder structure below.

- AntiUAV_train_val.zip

```
datasets/
├── train/*.jpg, *.txt
└── val/*.jpg, *.txt
```

- AntiUAV_train_val_test.zip

```
datasets/
├── test/*.jpg, *.txt
├── train/*.jpg, *.txt
└── val/*.jpg, *.txt
```


### (0) Official Dataset Release for the 4th Anti-UAV Challenge

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15103888.svg)](https://doi.org/10.5281/zenodo.15103888)

You can download the officially released datasets from the following links:

- Train (Track 1 & Track 2)
    - [Google Drive](https://drive.google.com/drive/folders/1hEGq14WnfPstYrI_9OgscR1VsWc5_XDl) | [Baidu](https://pan.baidu.com/s/1rtZ_PkYX__Bt2O5MgTj1tg?pwd=CVPR)
- Test (Track 1)
    - [Google Drive](https://drive.google.com/drive/folders/1qkUeglLk9-OXniIUVh1r7OljDLwDNhBs?usp=sharing) | [Baidu](https://pan.baidu.com/s/13HFq5P0gWrdlBerFZBKbuA?pwd=cvpr)
- Test (Track 2)
    - [Google Drive](https://drive.google.com/drive/folders/1qkUeglLk9-OXniIUVh1r7OljDLwDNhBs?usp=sharing) | [Baidu](https://pan.baidu.com/s/1s7KkyjgXP1v495EULqwoew?pwd=cvpr)

```
.
├── track1_test/
└── track2_test/
```