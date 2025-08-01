## YOLOv12-BoT-SORT-ReID




> [Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Chen_Strong_Baseline_Multi-UAV_Tracking_via_YOLOv12_with_BoT-SORT-ReID_CVPRW_2025_paper.html)
>
> Yu-Hsi Chen




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




<details><summary>Preface</summary>

The combination of YOLOv12 and BoT-SORT demonstrates strong object detection and tracking potential yet remains underexplored in current literature and implementations.

<img src="https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/assets/existing_methods_overview.png" width="100%">

```
[1] Jocher, Glenn, et al. "ultralytics/yolov5: v6. 0-YOLOv5n'Nano'models, Roboflow integration, TensorFlow export, OpenCV DNN support." Zenodo (2021).
[2] Tian, Yunjie, Qixiang Ye, and David Doermann. "Yolov12: Attention-centric real-time object detectors." arXiv preprint arXiv:2502.12524 (2025).
[3] Zhang, Guangdong, et al. "Multi-object Tracking Based on YOLOX and DeepSORT Algorithm." International Conference on 5G for Future Wireless Networks. Cham: Springer Nature Switzerland, 2022.
[4] Aharon, Nir, Roy Orfaig, and Ben-Zion Bobrovsky. "Bot-sort: Robust associations multi-pedestrian tracking." arXiv preprint arXiv:2206.14651 (2022).
```

</details>




This repository provides a strong baseline for multi-UAV tracking in thermal infrared videos by leveraging YOLOv12 and BoT-SORT with ReID. Our approach significantly outperforms the widely adopted YOLOv5 with the DeepSORT pipeline, offering a high-performance foundation for UAV swarm tracking. Importantly, the established workflow in this repository can be easily integrated with any custom-trained model, extending its applicability beyond UAV scenarios. Refer to [this](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID#-quickstart-installation-and-demonstration) section for practical usage examples.




<details><summary>📹 Preview - Strong Baseline</summary>

[strong_baseline.webm](https://github.com/user-attachments/assets/702b3e80-fd3c-48f0-8032-a2a97563c19f)

🔗 Full video available at: [Track 3](https://youtu.be/_IiUISzCeU8?si=19JnHdwS9GLoYdtL)

🔍 See also SOT inferences: [Track 1](https://youtu.be/HOwMRm1l124?si=ewlZ5wr1_CUDFWk_) and [Track 2](https://youtu.be/M7lSrqYkpEQ?si=EyVhfOPNRLPVzYI2)

🌐 [CVPR2025](https://cvpr.thecvf.com/) | [Workshops](https://cvpr.thecvf.com/Conferences/2025/workshop-list) | [4th Anti-UAV Workshop](https://anti-uav.github.io/) | [Track-1](https://codalab.lisn.upsaclay.fr/competitions/21688) | [Track-2](https://codalab.lisn.upsaclay.fr/competitions/21690) | [Track-3](https://codalab.lisn.upsaclay.fr/competitions/21806)

</details>




<details><summary>📹 Preview - Single-Frame Enhancements</summary>

[enhancements_MultiUAV-261.webm](https://github.com/user-attachments/assets/f1dd3877-d898-45c2-93c9-26f677020e07)

🔗 Full video available at: [Enhancements](https://youtu.be/lkIlYCjz8r4?si=7jpgs5OAEeABNVGo)

</details>




<details><summary>📹 Preview - Custom Model Inference</summary>

This section showcases example videos processed using a custom-trained model. The scenes are not limited to UAV footage or single-class detection. See [🚀 Quickstart: Installation and Demonstration](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID?tab=readme-ov-file#-quickstart-installation-and-demonstration) → `Run Inference Using a Custom-Trained Model` for more details.

<details><summary>1. Multi-Class on a Walkway Scene</summary>

[palace.webm](https://github.com/user-attachments/assets/cc32bda1-f461-4813-9639-eab2adfc178e)

🔗 Original video: [palace.mp4](https://github.com/FoundationVision/ByteTrack/blob/main/videos/palace.mp4)

</details>

<details><summary>2. Common Objects Underwater</summary>

[cou.webm](https://github.com/user-attachments/assets/59a81337-245a-49a7-817e-422536199b19)

🔗 Full video available at: [COU.mp4](https://youtu.be/dZAQnpDq7NQ?si=ovF637bp4D-HZ04_)

</details>

<details open><summary>3. UAVDB</summary>

[uavdb.webm](https://github.com/user-attachments/assets/3eff3e71-4111-4792-b4f6-4f1371843978)

🔗 Full video available at: [UAVDB.mp4](https://youtu.be/NOZ4yhgXF7Q?si=bPM0N3SjR6tcHH3z)

</details>

</details>








## 🏁 Beyond Strong Baseline: Multi-UAV Tracking Competition ₊˚⊹




<details><summary>📹 Preview - Vision in Action: Overview of All Videos</summary>

A complete visual overview of all training and test videos.

[vision_in_action.webm](https://github.com/user-attachments/assets/f50d8e90-63b8-4b62-84ca-7e71c0750c67)

🔗 Full video available at: [Overview](https://youtu.be/0-Sn_mxRPJw?si=xfFXvBNoQz8zxnbK)

Scenarios are categorized to evaluate tracking performance under diverse conditions:

- **Takeoff** - UAV launch phase: 2 videos.
- **L** - Larger UAV target: 15 videos.
- **C** - Cloud background: 39 videos.
- **CF** - Cloud (Fewer UAVs): 18 videos.
- **T** - Tree background: 68 videos.
- **TF** - Tree (Fewer UAVs): 14 videos.
- **B** - Scene with buildings: 11 videos.
- **BB1** - Building Background 1: 4 videos.
- **BB2** - Building Background 2: 17 videos.
- **BB2P** - Building Background 2 (UAV partially out of view): 8 videos.
- **Landing** - UAV landing phase: 4 videos.

**TOTAL: 200 videos (151,384 frames)**

</details>




<details><summary>📹 Preview - Vision in Action: Training Videos</summary>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15853476.svg)](https://doi.org/10.5281/zenodo.15853476)

[demo_train.webm](https://github.com/user-attachments/assets/e01c0bb5-f08e-4a76-829f-7d2ea717184e)

🔗 Full video available at: [Training Videos](https://youtu.be/rny0-nyFBk0?si=jxCPlCcHgU4zcUwU)

- **Takeoff** - UAV launch phase: 1 videos.
- **L** - Larger UAV target: 8 videos.
- **C** - Cloud background: 20 videos.
- **CF** - Cloud (Fewer UAVs): 9 videos.
- **T** - Tree background: 34 videos.
- **TF** - Tree (Fewer UAVs): 7 videos.
- **B** - Scene with buildings: 6 videos.
- **BB1** - Building Background 1: 2 videos.
- **BB2** - Building Background 2: 9 videos.
- **BB2P** - Building Background 2 (UAV partially out of view): 4 videos.
- **Landing** - UAV landing phase: 2 videos.

**TOTAL: 102 videos (77,293 frames)**

</details>




<details><summary>📹 Preview - Vision in Action: Test Videos</summary>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16299533.svg)](https://doi.org/10.5281/zenodo.16299533)

[demo_test.webm](https://github.com/user-attachments/assets/15e9143e-303f-4ef1-849d-735f8763e112)

🔗 Full video available at: [Test Videos](https://youtu.be/G_8fE9njTRs?si=xUJjaYYC3D81m3Na)

- **Takeoff** - UAV launch phase: 1 videos.
- **L** - Larger UAV target: 7 videos.
- **C** - Cloud background: 19 videos.
- **CF** - Cloud (Fewer UAVs): 9 videos.
- **T** - Tree background: 34 videos.
- **TF** - Tree (Fewer UAVs): 7 videos.
- **B** - Scene with buildings: 5 videos.
- **BB1** - Building Background 1: 2 videos.
- **BB2** - Building Background 2: 8 videos.
- **BB2P** - Building Background 2 (UAV partially out of view): 4 videos.
- **Landing** - UAV landing phase: 2 videos.

**TOTAL: 98 videos (74,538 frames)**

</details>




<details open><summary>📹 Preview - Vision in Action: Beyond Strong Baseline</summary>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16458805.svg)](https://doi.org/10.5281/zenodo.16458805)

[<img src="https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/assets/beyond_strong_baseline.png" width="100%">](https://www.codabench.org/competitions/9888/)

[<img src="https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/assets/beyond_strong_baseline_strong_baseline.png" width="100%">](https://www.codabench.org/competitions/9888/#/results-tab)

🔗 View the competition on [Codabench](https://www.codabench.org/competitions/9888/)

</details>








## 🗞️ News

- **August 1, 2025**: Submit now and challenge the [Strong Baseline](https://www.codabench.org/competitions/9888/#/results-tab) .
- **July 30, 2025**: [🔧 Corrected test data for the BB2P_02 sequence](https://doi.org/10.5281/zenodo.16601508) to fix a minor defect.
- **July 27, 2025**: [🏁 Beyond Strong Baseline](https://www.codabench.org/competitions/9888/) is now open for registration.
- **July 23, 2025**: The [test data](https://doi.org/10.5281/zenodo.16299533) for the competition is now available.
- **July 13, 2025**: The [training data](https://doi.org/10.5281/zenodo.15853476) for the competition is now available.
- **June 21, 2025**: Training scripts for [YOLOv12](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID#-reproduction) and [BoT-SORT-ReID](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID#-reproduction) are now available.
- **June 12, 2025**: 🥉 **3rd Place Award in The 4th Anti-UAV Workshop & Challenge Track 3**.
- **June 6, 2025**: Corrected mistyped numbers in [Table 1](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/tree/main/assets/Table_1.png) .
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
$ pip3 install seaborn
```

</details>


<details><summary>Folder Structure</summary>

The following folder structure will be created upon cloning this repository.

```
YOLOv12-BoT-SORT-ReID/
├── data/
│   └── demo/
│       ├── MOT/
│       │   ├── MultiUAV-003.mp4
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

Toy example with three tracks, including SOT and MOT.

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
$ python3 tools/predict_track3.py --weights ./yolov12/weights/MOT_yolov12n.pt --source ../data/demo/MOT/ --img-size 1600 --device "0" --track_buffer 60 --save_path_answer ./submit/track3/demo --with-reid --fast-reid-config logs/sbs_S50/config.yaml --fast-reid-weights logs/sbs_S50/model_0016.pth --hide-labels-name
# output: ./runs/detect/, ./submit/track3/demo/

# Heatmap
$ cd yolov12/
$ python heatmap.py
# output: ./outputs/
```

</details>


<details><summary>Run Inference on Custom Data</summary>

This project supports flexible inference on image folders and video files, with or without initial object positions, specifically for MOT task.

```bash
python3 tools/inference.py \
    --weights ./yolov12/weights/MOT_yolov12n.pt \
    --source <path to folder or video> \
    --with-initial-positions \
    --initial-position-config <path to initial positions file (optional)> \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name
```

Below are examples of supported inference settings:

```bash
# 1. Inference on Image Folder (without initial position)
python3 tools/inference.py \
    --weights ./yolov12/weights/MOT_yolov12n.pt \
    --source ../data/demo/MOT/Test_imgs/MultiUAV-003/ \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name

# 2. Inference on Image Folder (with initial position)
python3 tools/inference.py \
    --weights ./yolov12/weights/MOT_yolov12n.pt \
    --source ../data/demo/MOT/Test_imgs/MultiUAV-003/ \
    --with-initial-positions \
    --initial-position-config ../data/demo/MOT/TestLabels_FirstFrameOnly/MultiUAV-003.txt \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name

# 3. Inference on Video (without initial position)
python3 tools/inference.py \
    --weights ./yolov12/weights/MOT_yolov12n.pt \
    --source ../data/demo/MOT/MultiUAV-003.mp4 \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name

# 4. Inference on Video (with initial position)
python3 tools/inference.py \
    --weights ./yolov12/weights/MOT_yolov12n.pt \
    --source ../data/demo/MOT/MultiUAV-003.mp4 \
    --with-initial-positions \
    --initial-position-config ../data/demo/MOT/TestLabels_FirstFrameOnly/MultiUAV-003.txt \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name
```

</details>


<details><summary>Run Inference Using a Custom Trained Model</summary>

This project also supports flexible inference using a custom-trained model for any MOT task. Below are the instructions for reproducing the preview section.

```bash
$ cd BoT-SORT/
```

### 1. Multi-Class on a Walkway Scene

```bash
$ wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt
$ wget https://github.com/FoundationVision/ByteTrack/raw/main/videos/palace.mp4
$ python3 tools/inference.py \
    --weights yolov12x.pt \
    --source palace.mp4 \
    --img-size 640 \
    --device "0" \
    --save_path_answer ./submit/palace/
```

### 2. Common Objects Underwater

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15828323.svg)](https://doi.org/10.5281/zenodo.15828323)

```bash
for f in ./videos/COU/*.mp4; do
    python3 tools/inference.py \
        --weights ./yolov12/runs/det/train/weights/best.pt \
        --source "$f" \
        --img-size 1600 \
        --device "0" \
        --save_path_answer ./submit/COU/
done
```

### 3. UAVDB

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16342697.svg)](https://doi.org/10.5281/zenodo.16342697)

```bash
for f in ./videos/UAVDB/*.mp4; do
    python3 tools/inference.py \
        --weights ./yolov12/runs/det/train/weights/best.pt \
        --source "$f" \
        --img-size 1600 \
        --device "0" \
        --save_path_answer ./submit/UAVDB/
done
```

</details>








## 🛠️ Implementation Details


<details><summary>Hardware Information</summary>

Experiments were conducted on two platforms: (1) a local system with an Intel Core i7-12650H CPU, NVIDIA RTX 4050 GPU, and 16 GB RAM for data processing and inference, and (2) an HPC system with an NVIDIA H100 GPU and 80 GB memory for model training.

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
│   ├── getInfo.py
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

Executing the following commands can reproduce the leaderboard results.

<details><summary>Data Analysis</summary>

```bash
$ cd BoT-SORT/

# Table 1
$ python3 getInfo.py
```

</details>

<details><summary>Train YOLOv12</summary>

Refer to the [README](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/tree/main/data/MOT#readme) for more information.

```bash
$ cd BoT-SORT/yolov12/

# Run training with default settings
$ python3 train.py
```

</details>

<details><summary>Train BoT-SORT-ReID</summary>

Refer to the [README](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/tree/main/BoT-SORT/fast_reid/datasets#readme) for more information.

```bash
$ cd BoT-SORT/

# Train with final config
$ python3 fast_reid/tools/train_net.py --config-file ./logs/sbs_S50/config.yaml MODEL.DEVICE "cuda:0"
```

</details>

<details><summary>Inference</summary>

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

</details>








## ✨ Models

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/wish44165/YOLOv12-BoT-SORT-ReID) 

| Model                                                                                | size<br><sup>(pixels) | AP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(G) | Note |
| :----------------------------------------------------------------------------------- | :-------------------: | :-------------------:| :-----------------:| :---------------:| :----: |
| [SOT_yolov12l.pt](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/BoT-SORT/yolov12/weights/SOT_yolov12l.pt) | 640                   | 67.2                 | 26.3                | 88.5               |
| [MOT_yolov12n.pt](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/BoT-SORT/yolov12/weights/MOT_yolov12n.pt) ([ReID](https://huggingface.co/wish44165/YOLOv12-BoT-SORT-ReID/tree/main)) | 1600                   | 68.5                 | 2.6                | 6.3              | [#4 (Comment)](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/issues/4#issuecomment-2959336418) |








## 📜 Citation

If you find this project helpful for your research or applications, we would appreciate it if you could cite the paper and give it a star.

```
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Yu-Hsi},
    title     = {Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {6573-6582}
}
```

<a href="https://www.star-history.com/#wish44165/YOLOv12-BoT-SORT-ReID&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=wish44165/YOLOv12-BoT-SORT-ReID&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=wish44165/YOLOv12-BoT-SORT-ReID&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=wish44165/YOLOv12-BoT-SORT-ReID&type=Date" />
 </picture>
</a>








## 🙏 Acknowledgments

Much of the code builds upon [YOLOv12](https://github.com/sunsmarterjie/yolov12), [BoT-SORT](https://github.com/NirAharon/BoT-SORT), and [TrackEval](https://github.com/JonathonLuiten/TrackEval). We also sincerely thank the organizers of the [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) benchmark for providing the valuable dataset. We greatly appreciate their contributions!