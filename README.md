## Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID




[![arXiv](https://img.shields.io/badge/arXiv-2503.17237-b31b1b.svg)](https://arxiv.org/pdf/2503.17237)
[![PyPI - Python Version](https://img.shields.io/badge/python-3.11-blue.svg?logo=python&logoColor=gold)](https://www.python.org/downloads/release/python-3110/)
[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x5T6woUdV6dD_T6qdYcKG04Q2iVVHGoD?usp=sharing)




This repository provides a strong baseline for multi-UAV tracking in thermal infrared videos by leveraging YOLOv12 and BoT-SORT with ReID. Our method provides a significant boost over the well-established YOLOv5 with the DeepSORT combination, offering a high-performance starting point for UAV swarm tracking.




📹 Preview:

[strong_baseline.webm](https://github.com/user-attachments/assets/702b3e80-fd3c-48f0-8032-a2a97563c19f)

🔗 Full video available at: [Track 3](https://youtu.be/_IiUISzCeU8?si=19JnHdwS9GLoYdtL)

🔍 See also SOT inferences: [Track 1](https://youtu.be/HOwMRm1l124?si=ewlZ5wr1_CUDFWk_) and [Track 2](https://youtu.be/M7lSrqYkpEQ?si=EyVhfOPNRLPVzYI2)

🌐 [CVPR2025](https://cvpr.thecvf.com/) | [Workshops](https://cvpr.thecvf.com/Conferences/2025/workshop-list) | [4th Anti-UAV Workshop](https://anti-uav.github.io/) | [Track-1](https://codalab.lisn.upsaclay.fr/competitions/21688) | [Track-2](https://codalab.lisn.upsaclay.fr/competitions/21690) | [Track-3](https://codalab.lisn.upsaclay.fr/competitions/21806)








## 🗞️ News

Details soon








## 🚀 Quickstart: Installation and Demonstration


<details><summary>Installation</summary>

```bash
$ conda create -n yolov12_botsort python=3.11 -y
$ conda activate yolov12_botsort
$ git clone https://github.com/wish44165/YOLOv12-BoT-SORT-ReID.git
$ cd YOLOv12-BoT-SORT-ReID/BoT-SORT/yolov12/
$ wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
$ pip install -r requirements.txt
$ pip install -e .
$ cd ../
$ pip3 install torch torchvision torchaudio
$ pip3 install -r requirements.txt
$ python3 setup.py develop
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








## 🌟 Implementation Details


<details><summary>Hardware Information</summary>

### Laptop

- CPU: Intel® Core™ i7-12650H
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6GB)
- RAM: 23734MiB

### HPC

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
  
</details>


<details><summary>Strong Baseline</summary>

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


<details><summary>Enhancement</summary>

Details soon

</details>




### 📂 Folder Structure

Details soon




### 🔩 Reproduction

Details soon








## ✨ Models

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :----------------------------------------------------------------------------------- | :-------------------: | :-------------------:| :-----------------:| :---------------:|
| [SOT_yolov12l.pt](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/BoT-SORT/yolov12/weights/SOT_yolov12l.pt) | 640                   | 67.2                 | 26.3                | 88.5               |
| [MOT_yolov12n.pt](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/BoT-SORT/yolov12/weights/MOT_yolov12n.pt) | 1600                   | 77.7                 | 2.6                | 6.3              |








## 🧩 Integration

Details soon








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