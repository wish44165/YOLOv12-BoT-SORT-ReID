#!/bin/bash

# Set the base directory containing the test folders
SOURCE_DIR="../data/MOT/MultiUAV_Test/Test_imgs"

# Loop through each subdirectory in SOURCE_DIR
for folder in "$SOURCE_DIR"/*; do
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"
        python3 tools/predict_track3.py \
            --weights ./yolov12/weights/MOT_yolov12n.pt \
            --source "$folder" \
            --img-size 1600 \
            --track_buffer 60 \
            --device "0" \
            --agnostic-nms \
            --save_path_answer ./submit/track3/test/ \
            --with-reid \
            --fast-reid-config logs/sbs_S50/config.yaml \
            --fast-reid-weights logs/sbs_S50/model_0016.pth \
            --hide-labels-name
    fi
done

