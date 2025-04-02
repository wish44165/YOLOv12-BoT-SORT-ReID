import argparse
import time
import os
import json
from pathlib import Path
import sys
from ultralytics import YOLO
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# Adjust repository paths if needed
sys.path.insert(0, './yolov12')
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import check_img_size, scale_coords, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT

def process_folder(folder_path, model, tracker, device, half, imgsz, opt):
    """
    Process a single folder (subfolder of the source) containing images.
    Reads IR_label.json (if exists) for prior knowledge and then runs detection/tracking on each image.
    Returns a list of bounding boxes (one per frame) in [x, y, w, h] format.
    """
    # Check for IR_label.json in the folder
    gt_path = os.path.join(folder_path, "IR_label.json")
    prior_box = None
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
            if "gt_rect" in gt_data and len(gt_data["gt_rect"]) > 0:
                # Use the first bounding box as the prior knowledge
                prior_box = gt_data["gt_rect"][0]
                print(f"[{folder_path}] Loaded prior box from IR_label.json: {prior_box}")
    else:
        print(f"[{folder_path}] IR_label.json not found. No prior knowledge available.")

    # Initialize last_pred_box with the prior box (if any)
    last_pred_box = prior_box.copy() if prior_box is not None else None

    # Prepare dataset from the folder (assumes images only)
    dataset = LoadImages(folder_path, img_size=imgsz, stride=int(model.stride.max()))
    
    boxes_per_frame = []  # To record one box per frame

    for path, img, im0s, vid_cap in dataset:
        # Preprocess image
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        _ = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        from ultralytics.utils.ops import non_max_suppression
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)[0]
        _ = time_synchronized()

        # Process detections
        detections = []
        if pred is not None and pred.ndim:
            if pred.ndim == 1:  # single detection
                pred = pred.view(1, -1)
            if len(pred):
                boxes = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).cpu().numpy()
                detections = pred.cpu().numpy()
                detections[:, :4] = boxes

        online_targets = []
        if len(detections):
            online_targets = tracker.update(detections, im0s)

        # Select the best detection (if any) and update last_pred_box.
        if online_targets:
            best_target = max(online_targets, key=lambda t: t.score)
            tlwh = best_target.tlwh  # [x, y, w, h]
            last_pred_box = tlwh.copy()
            tlbr = best_target.tlbr  # [x1, y1, x2, y2]
            # Plot bounding box if required
            if opt.plot_and_save:
                plot_one_box(tlbr, im0s, label=f'{best_target.track_id}', 
                             color=[random.randint(0, 255) for _ in range(3)], line_thickness=2)
            boxes_per_frame.append(last_pred_box.tolist() if hasattr(last_pred_box, 'tolist') else last_pred_box)
        else:
            # No detection: use the last known bounding box if available.
            if last_pred_box is not None:
                boxes_per_frame.append(last_pred_box.tolist() if hasattr(last_pred_box, 'tolist') else last_pred_box)
                tlbr = [last_pred_box[0], last_pred_box[1], last_pred_box[0] + last_pred_box[2], last_pred_box[1] + last_pred_box[3]]
                if opt.plot_and_save:
                    plot_one_box(tlbr, im0s, label='pred', color=[0, 0, 255], line_thickness=2)
            else:
                boxes_per_frame.append([0, 0, 0, 0])
                print(f"Frame {path}: No detection and no prior bounding box available.")

    return boxes_per_frame

def main(opt):
    # Initialize logging, device, and model
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    
    # Load model
    model = attempt_load(opt.weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)
    if opt.trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=25.0)
    
    # Warm up the model
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # Determine output directory based on --save_path_answer.
    if opt.save_path_answer is not None:
        output_dir = opt.save_path_answer
    elif os.path.isdir(opt.source):
        output_dir = opt.source.rstrip('/\\') + "_label"
    else:
        output_dir = str(Path(opt.project) / (Path(opt.source).stem + "_label"))
    os.makedirs(output_dir, exist_ok=True)

    # If the source is a directory containing subfolders:
    if os.path.isdir(opt.source):
        for sub in os.listdir(opt.source):
            sub_path = os.path.join(opt.source, sub)
            if os.path.isdir(sub_path):
                print(f"Processing folder: {sub_path}")
                boxes_list = process_folder(sub_path, model, tracker, device, half, imgsz, opt)
                json_str = json.dumps({"res": boxes_list}, separators=(',', ':'))
                output_file = os.path.join(output_dir, f"{sub}.txt")
                with open(output_file, "w") as f:
                    f.write(json_str)
                print(f"Saved labels for {sub} to {output_file}")
    else:
        print(f"Processing source: {opt.source}")
        boxes_list = process_folder(opt.source, model, tracker, device, half, imgsz, opt)
        json_str = json.dumps({"res": boxes_list}, separators=(',', ':'))
        output_file = Path(output_dir) / (Path(opt.source).stem + ".txt")
        os.makedirs(output_file.parent, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(json_str)
        print(f"Saved labels to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov12.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source directory (can contain subfolders)')
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels); should match training')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # Tracking arguments
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes with high aspect ratio")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true", help="fuse score and iou for association")

    # CMC & ReID parameters (if needed)
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid weights file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    # Additional arguments
    parser.add_argument('--save_path_answer', type=str, default=None, help='Path to save the label files. If not set, "_label" is appended to source.')
    parser.add_argument('--plot_and_save', action='store_true', help='If set, plot bounding boxes on images and save the output images.')

    opt = parser.parse_args()
    opt.jde = False
    opt.ablation = False

    print(opt)
    with torch.no_grad():
        if opt.update:
            for opt.weights in Path(opt.weights).expanduser().glob('*.pt'):
                strip_optimizer(opt.weights)
        main(opt)

