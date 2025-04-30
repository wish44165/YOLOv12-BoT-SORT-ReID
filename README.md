## Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID




[![arXiv](https://img.shields.io/badge/arXiv-2503.17237-b31b1b.svg)](https://arxiv.org/abs/2503.17237)
[![PyPI - Python Version](https://img.shields.io/badge/python-3.11-blue.svg?logo=python&logoColor=gold)](https://www.python.org/downloads/release/python-3110/)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wish44165/YOLOv12-BoT-SORT-ReID) 
[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x5T6woUdV6dD_T6qdYcKG04Q2iVVHGoD?usp=sharing)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/yuhsi44165/yolov12-bot-sort/)

<a href="https://doi.org/10.5281/zenodo.15203123"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15203123.svg" alt="DOI"></a>
<a href="https://github.com/wish44165/wish44165/tree/main/assets"><img src="https://github.com/wish44165/wish44165/blob/main/assets/msi_Cyborg_15_A12VE_badge.svg" alt="MSI"></a> 
<a href="https://dashboard.hpc.unimelb.edu.au/"><img src="https://github.com/wish44165/wish44165/blob/main/assets/unimelb_spartan.svg" alt="Spartan"></a> 

[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/publication/390114692_Strong_Baseline_Multi-UAV_Tracking_via_YOLOv12_with_BoT-SORT-ReID)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@scofield44165/strong-baseline-multi-uav-tracking-via-yolov12-with-bot-sort-reid-5d6b71230e39)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/playlist?list=PLfr5E6mAx5EUpqP41CPSm5Nwfqe35iwtl)




This repository provides a strong baseline for multi-UAV tracking in thermal infrared videos by leveraging YOLOv12 and BoT-SORT with ReID. Our method provides a significant boost over the well-established YOLOv5 with the DeepSORT combination, offering a high-performance starting point for UAV swarm tracking.




📹 Preview - Strong Baseline

[strong_baseline.webm](https://github.com/user-attachments/assets/702b3e80-fd3c-48f0-8032-a2a97563c19f)

🔗 Full video available at: [Track 3](https://youtu.be/_IiUISzCeU8?si=19JnHdwS9GLoYdtL)

🔍 See also SOT inferences: [Track 1](https://youtu.be/HOwMRm1l124?si=ewlZ5wr1_CUDFWk_) and [Track 2](https://youtu.be/M7lSrqYkpEQ?si=EyVhfOPNRLPVzYI2)

🌐 [CVPR2025](https://cvpr.thecvf.com/) | [Workshops](https://cvpr.thecvf.com/Conferences/2025/workshop-list) | [4th Anti-UAV Workshop](https://anti-uav.github.io/) | [Track-1](https://codalab.lisn.upsaclay.fr/competitions/21688) | [Track-2](https://codalab.lisn.upsaclay.fr/competitions/21690) | [Track-3](https://codalab.lisn.upsaclay.fr/competitions/21806)




---




📹 Preview - Single-Frame Enhancements

[enhancements_MultiUAV-261.webm](https://github.com/user-attachments/assets/f1dd3877-d898-45c2-93c9-26f677020e07)

🔗 Full video available at: [Enhancements](https://youtu.be/lkIlYCjz8r4?si=7jpgs5OAEeABNVGo)




## 🗞️ News

- **April 25, 2025**: Single-Frame Enhancement [datasets](https://doi.org/10.5281/zenodo.15276582) are now available.
- **April 23, 2025**: Strong Baseline weights available: [YOLOv12](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/tree/main/BoT-SORT/yolov12/weights) | [ReID](https://huggingface.co/wish44165/YOLOv12-BoT-SORT-ReID/tree/main) .
- **April 13, 2025**: The [datasets](https://doi.org/10.5281/zenodo.15203123) presented in Table 2 of the [paper](https://arxiv.org/pdf/2503.17237) are now available.
- **April 7, 2025**: Our paper is now on [arXiv](https://arxiv.org/abs/2503.17237) .
    - 🎥 Demos: [Hugging Face](https://huggingface.co/spaces/wish44165/YOLOv12-BoT-SORT-ReID) | [YouTube](https://www.youtube.com/playlist?list=PLfr5E6mAx5EUpqP41CPSm5Nwfqe35iwtl)  
    - 🚀 Quickstart: [Colab Notebook](https://colab.research.google.com/drive/1x5T6woUdV6dD_T6qdYcKG04Q2iVVHGoD?usp=sharing) | [Kaggle Notebook](https://www.kaggle.com/code/yuhsi44165/yolov12-bot-sort/)  
    - 🌐 Project Page: [Link](https://sites.google.com/view/wish44165/home/academic-activities/2025/strong-baseline-multi-uav-tracking-via-yolov12-with-bot-sort-reid)








## 🚀 Quickstart: Installation and Demonstration

[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x5T6woUdV6dD_T6qdYcKG04Q2iVVHGoD?usp=sharing)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/yuhsi44165/yolov12-bot-sort/)

<details><summary>Installation</summary>

```bash
$ conda create -n yolov12_botsort python=3.11 -y
$ conda activate yolov12_botsort
$ git clone https://github.com/wish44165/YOLOv12-BoT-SORT-ReID.git
$ cd YOLOv12-BoT-SORT-ReID/BoT-SORT/yolov12/
$ wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
$ pip install -r requirements.txt
$ cd ../
$ pip3 install torch torchvision torchaudio
$ pip3 install -r requirements.txt
$ pip3 install ultralytics
$ pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
$ pip3 install cython_bbox
$ pip3 install faiss-cpu
```

</details>


<details><summary>Folder Structure</summary>

```
YOLOv12-BoT-SORT-ReID/
├── data/
│   └── demo/
│       ├── MOT/
│       │   ├── Test_imgs/
│       │   │   ├── MultiUAV-003/
│       │   │   ├── MultiUAV-135/
│       │   │   ├── MultiUAV-173/
│       │   │   └── MultiUAV-261/
│       │   └── TestLabels_FirstFrameOnly/
│       │       ├── MultiUAV-003.txt
│       │       ├── MultiUAV-135.txt
│       │       ├── MultiUAV-173.txt
│       │       └── MultiUAV-261.txt
│       └── SOT/
│           ├── Track1/
│           │   ├── 20190926_111509_1_8/
│           │   ├── 41_1/
│           │   ├── new30_train-new/
│           │   └── wg2022_ir_050_split_01/
│           └── Track2/
│               ├── 02_6319_0000-1499/
│               ├── 3700000000002_110743_1/
│               ├── DJI_0057_1/
│               └── wg2022_ir_032_split_04/
└── BoT-SORT/
```

</details>


<details><summary>Demonstration</summary>

```bash
$ cd BoT-SORT/

# Track 1
$ python3 tools/predict_track1.py --weights ./yolov12/weights/SOT_yolov12l.pt --source ../data/demo/SOT/Track1/ --img-size 640 --device "0" --conf-thres 0.01 --iou-thres 0.01 --track_high_thresh 0.1 --track_low_thresh 0.01 --fuse-score --agnostic-nms --min_box_area 4 --save_path_answer ./submit/track1/demo --hide-labels-name
# output: ./runs/detect/, ./submit/track1/demo/

# Track 2
$ python3 tools/predict_track2.py --weights ./yolov12/weights/SOT_yolov12l.pt --source ../data/demo/SOT/Track2/ --img-size 640 --device "0" --conf-thres 0.01 --iou-thres 0.01 --track_high_thresh 0.1 --track_low_thresh 0.01 --fuse-score --agnostic-nms --min_box_area 1 --save_path_answer ./submit/track2/demo --hide-labels-name
# output: ./runs/detect/, ./submit/track2/demo/

# Track 3
$ python3 tools/predict_track3.py --weights ./yolov12/weights/MOT_yolov12n.pt --source ../data/demo/MOT/ --img-size 1600 --device "0" --track_buffer 60 --save_path_answer ./submit/track3/demo --hide-labels-name
# output: ./runs/detect/, ./submit/track3/demo/

# Heatmap
$ cd yolov12/
$ python heatmap.py
# output: ./outputs/
```

</details>








## 🛠️ Implementation Details


<details><summary>Hardware Information</summary>

### Laptop

<a href="https://github.com/wish44165/wish44165/tree/main/assets"><img src="https://github.com/wish44165/wish44165/blob/main/assets/msi_Cyborg_15_A12VE_badge.svg" alt="Spartan"></a> 

- CPU: Intel® Core™ i7-12650H
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6GB)
- RAM: 23734MiB

### HPC

<a href="https://dashboard.hpc.unimelb.edu.au/"><img src="https://github.com/wish44165/wish44165/blob/main/assets/unimelb_spartan.svg" alt="Spartan"></a> 

- GPU: Spartan gpu-h100 (80GB), gpu-a100 (80GB)
  
</details>




### 🖻 Data Preparation


<details><summary>Officially Released</summary>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15103888.svg)](https://doi.org/10.5281/zenodo.15103888)

```
4th_Anti-UAV_Challenge/
├── baseline/
│   ├── Baseline_code.zip
│   └── MultiUAV_Baseline_code_and_submissi.zip
├── test/
│   ├── MultiUAV_Test.zip
│   ├── track1_test.zip
│   └── track2_test.zip
└── train/
    ├── MultiUAV_Train.zip
    └── train.zip
```

- Train
    - Track 1 & Track 2: [Google Drive](https://drive.google.com/drive/folders/1hEGq14WnfPstYrI_9OgscR1VsWc5_XDl) | [Baidu](https://pan.baidu.com/s/1rtZ_PkYX__Bt2O5MgTj1tg?pwd=CVPR)
    - Track 3: [Google Drive](https://drive.google.com/drive/folders/1JvGdAJjGzjOIGMG82Qiz5YJKzjy8VOd-?usp=drive_link) | [Baidu](https://pan.baidu.com/s/19iVwI1MW9OdXyPIc0xBSjQ?from=init&pwd=CVPR)
- Test
    - Track 1: [Google Drive](https://drive.google.com/drive/folders/1qkUeglLk9-OXniIUVh1r7OljDLwDNhBs?usp=sharing) | [Baidu](https://pan.baidu.com/s/13HFq5P0gWrdlBerFZBKbuA?pwd=cvpr)
    - Track 2: [Google Drive](https://drive.google.com/drive/folders/1qkUeglLk9-OXniIUVh1r7OljDLwDNhBs?usp=sharing) | [Baidu](https://pan.baidu.com/s/1s7KkyjgXP1v495EULqwoew?pwd=cvpr)
    - Track 3: [Google Drive](https://drive.google.com/drive/folders/1cfF00w_3ewUMELSSnmaYOKLTZoIWlxbF?usp=sharing) | [Baidu](https://pan.baidu.com/s/1rhB24tksTw1JW6ZltOSvOg?pwd=CVPR)

</details>


<details><summary>Strong Baseline</summary>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15203123.svg)](https://doi.org/10.5281/zenodo.15203123)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/wish44165/StrongBaseline_YOLOv12-BoT-SORT-ReID) 

```
train/
├── MOT/
│   └── AntiUAV_train_val.zip
├── ReID/
│   ├── MOT20_subset.zip
│   └── MOT20.zip
└── SOT/
    ├── AntiUAV_train_val_test.zip
    └── AntiUAV_train_val.zip
```

</details>


<details><summary>Enhancements</summary>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15276582.svg)](https://doi.org/10.5281/zenodo.15276582)

```
enhancements/
├── MOT/
│   ├── CLAHE_train_val.zip
│   ├── Sobel-based_Edge_Sharpening_train_val.zip
│   └── Sobel-based_Image_Gradients_train_val.zip
└── ReID/
    ├── CLAHE_subset.zip
    ├── Sobel-based_Edge_Sharpening_subset.zip
    └── Sobel-based_Image_Gradients_subset.zip
```

</details>




### 📂 Folder Structure

<details><summary>Project Layout</summary>

Follow the folder structure below to ensure smooth execution and easy navigation.

```
YOLOv12-BoT-SORT-ReID/
├── BoT-SORT/
│   ├── datasets/
│   │   └── README.md
│   ├── fast_reid/
│   │   └── datasets/
│   │       ├── generate_mot_patches.py
│   │       └── README.md
│   ├── logs/
│   │   ├── sbs_S50/
│   │   │   ├── config.yaml
│   │   │   └── model_0016.pth
│   │   └── README.md
│   ├── requirements.txt
│   ├── runs/
│   │   └── README.md
│   ├── submit/
│   │   └── README.md
│   ├── tools/
│   │   ├── predict_track1.py
│   │   ├── predict_track2.py
│   │   └── predict_track3.py
│   └── yolov12/
│       ├── heatmap.py
│       ├── imgs_dir/
│       │   ├── 00096.jpg
│       │   ├── 00379.jpg
│       │   ├── 00589.jpg
│       │   └── 00643.jpg
│       ├── requirements.txt
│       └── weights/
│           ├── MOT_yolov12n.pt
│           └── SOT_yolov12l.pt
├── data/
│   ├── demo/
│   ├── MOT/
│   │   └── README.md
│   └── SOT/
│       └── README.md
├── LICENSE
└── README.md
```

</details>




### 🔨 Reproduction

<details><summary>Run Commands</summary>

```bash
$ cd BoT-SORT/

# Track 1
$ python3 tools/predict_track1.py --weights ./yolov12/weights/SOT_yolov12l.pt --source ../data/SOT/track1_test/ --img-size 640 --device "0" --conf-thres 0.01 --iou-thres 0.01 --track_high_thresh 0.1 --track_low_thresh 0.01 --fuse-score --agnostic-nms --min_box_area 4 --save_path_answer ./submit/track1/test --hide-labels-name
# output: ./runs/detect/, ./submit/track1/test/

# Track 2
$ python3 tools/predict_track2.py --weights ./yolov12/weights/SOT_yolov12l.pt --source ../data/SOT/track2_test/ --img-size 640 --device "0" --conf-thres 0.01 --iou-thres 0.01 --track_high_thresh 0.1 --track_low_thresh 0.01 --fuse-score --agnostic-nms --min_box_area 1 --save_path_answer ./submit/track2/test --hide-labels-name
# output: ./runs/detect/, ./submit/track2/test/

# Track 3
$ chmod +x run_track3.sh
$ ./run_track3.sh
# output: ./runs/detect/, ./submit/track3/test/
```

</details>








## ✨ Models

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/wish44165/YOLOv12-BoT-SORT-ReID) 

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :----------------------------------------------------------------------------------- | :-------------------: | :-------------------:| :-----------------:| :---------------:|
| [SOT_yolov12l.pt](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/BoT-SORT/yolov12/weights/SOT_yolov12l.pt) | 640                   | 67.2                 | 26.3                | 88.5               |
| [MOT_yolov12n.pt](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/BoT-SORT/yolov12/weights/MOT_yolov12n.pt) ([ReID](https://huggingface.co/wish44165/YOLOv12-BoT-SORT-ReID/tree/main/sbs_S50)) | 1600                   | 77.7                 | 2.6                | 6.3              |








## 📜 Citation

If you find this project helpful for your research or applications, we would appreciate it if you could give it a star and cite the paper.

```
@article{chen2025strong,
  title={Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID},
  author={Chen, Yu-Hsi},
  journal={arXiv preprint arXiv:2503.17237},
  year={2025}
}
```








## 🙏 Acknowledgments

A large portion of the code is adapted from [YOLOv12](https://github.com/sunsmarterjie/yolov12) and [BoT-SORT](https://github.com/NirAharon/BoT-SORT). Thanks for their excellent work!