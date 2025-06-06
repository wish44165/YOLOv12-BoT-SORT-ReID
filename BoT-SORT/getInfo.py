import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict




def sot_train(base_folder):
    heights, widths, areas = [], [], []
    total_boxes = 0
    removed_non_exist = 0
    removed_zero_size = 0
    removed_full_image = 0
    skipped_folders = 0

    seq_count_512 = 0
    seq_count_640 = 0
    frame_count_512 = 0
    frame_count_640 = 0

    for subfolder in sorted(os.listdir(base_folder)):
        seq_path = os.path.join(base_folder, subfolder)
        json_path = os.path.join(seq_path, "IR_label.json")

        if not os.path.isdir(seq_path) or not os.path.isfile(json_path):
            skipped_folders += 1
            continue

        img_files = sorted([f for f in os.listdir(seq_path) if f.lower().endswith((".jpg", ".png"))])
        if len(img_files) == 0:
            continue

        first_img_path = os.path.join(seq_path, img_files[0])
        try:
            with Image.open(first_img_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"Failed to read {first_img_path}: {e}")
            continue

        if (w, h) == (512, 512):
            seq_count_512 += 1
            frame_count_512 += len(img_files)
        elif (w, h) == (640, 512):
            seq_count_640 += 1
            frame_count_640 += len(img_files)
        else:
            print(f"Warning: Unexpected resolution {w}x{h} in {subfolder}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        exists = data.get("exist", [])
        boxes = data.get("gt_rect", [])

        for i, box in enumerate(boxes):
            if i < len(exists) and exists[i] == 1:
                if len(box) == 4:
                    x, y, bw, bh = box

                    if bw == w and bh == h:
                        removed_full_image += 1
                        continue
                    if bw == 0 or bh == 0:
                        removed_zero_size += 1
                        continue

                    widths.append(bw)
                    heights.append(bh)
                    areas.append(bw * bh)
                    total_boxes += 1
            else:
                removed_non_exist += 1

    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)

    print("====== Dataset Summary ======")
    print(f"Number of Sequences: {seq_count_512 + seq_count_640}")
    print(f"Number of Frames: {frame_count_512 + frame_count_640}")
    resolution_stats = {
        "512x512": {"seqs": seq_count_512, "frames": frame_count_512},
        "640x512": {"seqs": seq_count_640, "frames": frame_count_640},
    }
    print("Resolutions:")
    for res, stat in resolution_stats.items():
        print(f"  {res} ({stat['seqs']} seqs, {stat['frames']} frames)")

    print("\n====== Bounding Box Statistics ======")
    print(f"Total Bounding Boxes: {total_boxes}")
    if total_boxes > 0:
        print(f"Width Range (px): [{np.min(widths):.2f}, {np.max(widths):.2f}]")
        print(f"Width Mean ± Std (px): {np.mean(widths):.2f} ± {np.std(widths):.2f}")
        print(f"Height Range (px): [{np.min(heights):.2f}, {np.max(heights):.2f}]")
        print(f"Height Mean ± Std (px): {np.mean(heights):.2f} ± {np.std(heights):.2f}")
        print(f"Area Range (px²): [{np.min(areas):.2f}, {np.max(areas):.2f}]")
        print(f"Area Mean ± Std (px²): {np.mean(areas):.2f} ± {np.std(areas):.2f}")
    else:
        print("No valid bounding boxes found.")

    print("\n====== Removed Boxes Summary ======")
    print(f"Removed due to non-existence: {removed_non_exist}")
    print(f"Removed due to zero size: {removed_zero_size}")
    print(f"Removed due to full image size: {removed_full_image}")
    print(f"Skipped folders (no label/img): {skipped_folders}")

    if total_boxes > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, arr, label in zip(axes, [widths, heights, areas], ["Width", "Height", "Area"]):
            ax.hist(arr, bins=30, color='skyblue', edgecolor='black')
            ax.set_title(f"{label} Distribution")
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()




def mot(base_folder, video_folder):
    heights, widths, areas = [], [], []
    total_boxes = 0
    removed_zero_size = 0
    removed_full_image = 0
    num_objects_per_frame = []

    resolution_stats = {}
    num_sequences = 0
    total_frames = 0

    for txt_file in os.listdir(base_folder):
        if not txt_file.endswith('.txt'):
            continue

        txt_path = os.path.join(base_folder, txt_file)
        video_name = txt_file.replace('.txt', '.mp4')
        video_path = os.path.join(video_folder, video_name)

        img_w, img_h = None, None
        frame_count = 0

        if os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

        if img_w is None or img_h is None or frame_count == 0:
            continue

        res_key = f"{img_w}x{img_h}"
        if res_key not in resolution_stats:
            resolution_stats[res_key] = {"seqs": 0, "frames": 0}
        resolution_stats[res_key]["seqs"] += 1
        resolution_stats[res_key]["frames"] += frame_count

        total_frames += frame_count
        num_sequences += 1

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 9:
                frame_id, obj_id, x, y, w, h, _, _, _ = map(float, parts)
                if w == img_w and h == img_h:
                    removed_full_image += 1
                    continue
                if w == 0 or h == 0:
                    removed_zero_size += 1
                    continue
                boxes.append([x, y, w, h])

        num_objects_per_frame.append(len(boxes))

        for box in boxes:
            x, y, w, h = box
            heights.append(h)
            widths.append(w)
            areas.append(w * h)
            total_boxes += 1

    heights, widths, areas = map(np.array, [heights, widths, areas])
    num_objects_per_frame = np.array(num_objects_per_frame)

    print("====== Dataset Summary ======")
    print(f"Number of Sequences: {num_sequences}")
    print(f"Number of Frames: {total_frames}")
    print("Resolutions:")
    for res, stats in reversed(resolution_stats.items()):
        print(f"  {res} ({stats['seqs']} seqs, {stats['frames']} frames)")

    print("\n====== Bounding Box Statistics ======")
    print(f"Total Bounding Boxes: {total_boxes}")
    if total_boxes > 0:
        print(f"Width Range (px): [{np.min(widths):.2f}, {np.max(widths):.2f}]")
        print(f"Width Mean ± Std (px): {np.mean(widths):.2f} ± {np.std(widths):.2f}")
        print(f"Height Range (px): [{np.min(heights):.2f}, {np.max(heights):.2f}]")
        print(f"Height Mean ± Std (px): {np.mean(heights):.2f} ± {np.std(heights):.2f}")
        print(f"Area Range (px²): [{np.min(areas):.2f}, {np.max(areas):.2f}]")
        print(f"Area Mean ± Std (px²): {np.mean(areas):.2f} ± {np.std(areas):.2f}")
        # print(f"Provided Bounding Boxes per Sequence: [{np.min(num_objects_per_frame)}, {np.max(num_objects_per_frame)}], Mean ± Std: {np.mean(num_objects_per_frame):.2f} ± {np.std(num_objects_per_frame):.2f}")
    else:
        print("No valid bounding boxes found.")

    print("\n====== Removed Boxes Summary ======")
    print(f"Removed due to zero size: {removed_zero_size}")
    print(f"Removed due to full image size: {removed_full_image}")

    if total_boxes > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, data, label in zip(axes, [heights, widths, areas], ["Height", "Width", "Area"]):
            ax.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(f"{label} Distribution")
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")

        plt.figure(figsize=(8, 6))
        plt.hist(num_objects_per_frame, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("Number of Objects per Frame Distribution")
        plt.xlabel("Number of Objects")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()




def sot_test(base_folder):
    heights, widths, areas = [], [], []
    total_boxes = 0
    removed_zero_size = 0
    removed_full_image = 0
    total_frames = 0
    seq_count = 0
    objs_per_frame = []

    resolution_stats = defaultdict(lambda: {'seqs': 0, 'frames': 0})

    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        seq_count += 1

        image_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        total_frames += len(image_files)

        if not image_files:
            continue

        try:
            first_img_path = os.path.join(subfolder_path, image_files[0])
            with Image.open(first_img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"Failed to open first image in {subfolder_path}: {e}")
            continue

        res_key = f"{img_w}x{img_h}"
        resolution_stats[res_key]['seqs'] += 1
        resolution_stats[res_key]['frames'] += len(image_files)

        json_path = os.path.join(subfolder_path, "IR_label.json")
        if not os.path.isfile(json_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        boxes = data.get("gt_rect", [])
        frame_obj_counts = []

        for i, frame_boxes in enumerate(boxes):
            count = 0
            if i >= len(image_files):
                break

            img_path = os.path.join(subfolder_path, image_files[i])
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"Failed to open image {img_path}: {e}")
                continue

            frame_boxes_list = [frame_boxes] if isinstance(frame_boxes[0], (int, float)) else frame_boxes
            for box in frame_boxes_list:
                if len(box) == 4:
                    x, y, w, h = box
                    if w == img_w and h == img_h:
                        removed_full_image += 1
                        continue
                    if w == 0 or h == 0:
                        removed_zero_size += 1
                        continue
                    widths.append(w)
                    heights.append(h)
                    areas.append(w * h)
                    total_boxes += 1
                    count += 1
            frame_obj_counts.append(count)

        objs_per_frame.extend(frame_obj_counts)
    
    print("====== Dataset Summary ======")
    print(f"Number of Sequences: {seq_count}")
    print(f"Number of Frames: {total_frames}")
    print("Resolutions:")
    for res, stats in reversed(resolution_stats.items()):
        print(f"  {res} ({stats['seqs']} seqs, {stats['frames']} frames)")

    print("\n====== Bounding Box Statistics ======")
    if heights:
        heights = np.array(heights)
        widths = np.array(widths)
        areas = np.array(areas)
        objs_per_frame = np.array(objs_per_frame)

        print(f"Total Bounding Boxes: {total_boxes}")
        print(f"Width Range (px): [{np.min(widths):.2f}, {np.max(widths):.2f}]")
        print(f"Width Mean ± Std (px): {np.mean(widths):.2f} ± {np.std(widths):.2f}")
        print(f"Height Range (px): [{np.min(heights):.2f}, {np.max(heights):.2f}]")
        print(f"Height Mean ± Std (px): {np.mean(heights):.2f} ± {np.std(heights):.2f}")
        print(f"Area Range (px²): [{np.min(areas):.2f}, {np.max(areas):.2f}]")
        print(f"Area Mean ± Std (px²): {np.mean(areas):.2f} ± {np.std(areas):.2f}")
        # print(f"Provided Bounding Boxes per Sequence: [{np.min(objs_per_frame)}, {np.max(objs_per_frame)}], Mean ± Std: {np.mean(objs_per_frame):.2f} ± {np.std(objs_per_frame):.2f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, data, label in zip(axes, [heights, widths, areas], ["Height", "Width", "Area"]):
            ax.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(f"{label} Distribution")
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    else:
        print("No valid bounding boxes found.")

    print("\n====== Removed Boxes Summary ======")
    print(f"Removed due to zero size: {removed_zero_size}")
    print(f"Removed due to full image size: {removed_full_image}")




if __name__ == "__main__":
    print('='*40 + ' SOT Train ' + '='*40)
    sot_train("../train")
    
    print('\n\n' + '='*40 + ' MOT Train ' + '='*40)
    mot("../MultiUAV_Train/TrainLabels", "../MultiUAV_Train/TrainVideos")
    
    print('\n\n' + '='*40 + ' Track 1 Test ' + '='*40)
    sot_test("../test/track1_test/")
    
    print('\n\n' + '='*40 + ' Track 2 Test ' + '='*40)
    sot_test("../test/track2_test/")
    
    print('\n\n' + '='*40 + ' Track 3 Test ' + '='*40)
    mot("../test/MultiUAV_Test/TestLabels_FirstFrameOnly", "../test/MultiUAV_Test/TestVideos")