## Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID




This [paper](https://arxiv.org/abs/2503.17237) establishes a strong baseline for multi-UAV tracking in thermal infrared videos by leveraging YOLOv12 and BoT-SORT with ReID. Our method provides a significant boost over the well-established YOLOv5 with the DeepSORT combination, offering a high-performance starting point for UAV swarm tracking.

📹 Preview:

[strong_baseline.webm](https://github.com/user-attachments/assets/702b3e80-fd3c-48f0-8032-a2a97563c19f)

🔗 Full video available at: [MOT](https://youtu.be/_IiUISzCeU8?si=19JnHdwS9GLoYdtL)

🔍 See also SOT inferences: [Track-1](https://youtu.be/HOwMRm1l124?si=ewlZ5wr1_CUDFWk_) and [Track-2](https://youtu.be/M7lSrqYkpEQ?si=EyVhfOPNRLPVzYI2)


[CVPR2025](https://cvpr.thecvf.com/) | [4th Anti-UAV Workshop](https://anti-uav.github.io/) | [Track 1](https://codalab.lisn.upsaclay.fr/competitions/21688) | [Track 2](https://codalab.lisn.upsaclay.fr/competitions/21690) | [Track 3](https://codalab.lisn.upsaclay.fr/competitions/21806)




## Quickstart: Installation and Demonstration

<details><summary>Installation</summary>

```bash
$ conda create -n yolov12_botsort python=3.11 -y
$ conda activate yolov12_botsort
$ git clone https://github.com/wish44165/YOLOv12-BoT-SORT-ReID.git
$ cd YOLOv12-BoT-SORT-ReID/yolov12/
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
├── data
│   ├── demo
│   │   ├── MOT
│   │   │   ├── Test_imgs
│   │   │   │   ├── MultiUAV-003
│   │   │   │   ├── MultiUAV-135
│   │   │   │   ├── MultiUAV-173
│   │   │   │   └── MultiUAV-261
│   │   │   └── TestLabels_FirstFrameOnly
│   │   │       ├── MultiUAV-003.txt
│   │   │       ├── MultiUAV-135.txt
│   │   │       ├── MultiUAV-173.txt
│   │   │       └── MultiUAV-261.txt
│   │   └── SOT
│   │       ├── Track1
│   │       │   ├── 20190926_111509_1_8
│   │       │   ├── 41_1
│   │       │   ├── new30_train-new
│   │       │   └── wg2022_ir_050_split_01
│   │       └── Track2
│   │           ├── 02_6319_0000-1499
│   │           ├── 3700000000002_110743_1
│   │           ├── DJI_0057_1
│   │           └── wg2022_ir_032_split_04
│   ├── test
│   │   ├── MOT
│   │   │   └── README.md
│   │   └── SOT
│   │       └── README.md
│   └── train
│       ├── MOT
│       │   └── README.md
│       └── SOT
│           └── README.md
└── README.md
```
  
</details>


<details><summary>Demonstration</summary>

Details soon
  
</details>




## Details Implementation


<details><summary>Hardware Information</summary>

### Laptop

- CPU: Intel® Core™ i7-12650H
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6GB)
- RAM: 23734MiB

### HPC

- GPU: Spartan gpu-h100 (80GB), gpu-a100 (80GB)
  
</details>


Details soon




## Citation

If you find this project helpful for your research or applications, we would appreciate it if you could give it a star and cite the paper.

```
@article{chen2025strong,
  title={Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID},
  author={Chen, Yu-Hsi},
  journal={arXiv preprint arXiv:2503.17237},
  year={2025}
}
```




## Acknowledgments

A large portion of the code is adapted from [YOLOv12](https://github.com/sunsmarterjie/yolov12) and [BoT-SORT](https://github.com/NirAharon/BoT-SORT). We greatly appreciate their excellent work!