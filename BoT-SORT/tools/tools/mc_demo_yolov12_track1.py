import argparse
import time
from pathlib import Path
import sys
import os
import json
from ultralytics import YOLO
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

sys.path.insert(0, './yolov12')  # Adjust path to YOLOv12 repo

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, apply_classifier, scale_coords, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

sys.path.insert(0, './yolov7')  # Adjust path to YOLOv12 repo
sys.path.append('.')

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                          w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('Saved results to {}'.format(filename))

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Prior Knowledge: Load ground truth from IR_label.json (expected format: {"gt_rect": [[x, y, w, h]]})
    gt_path = os.path.join(source, "IR_label.json")
    prior_box = None
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
            if "gt_rect" in gt_data and len(gt_data["gt_rect"]) > 0:
                # Use the first bounding box as the prior knowledge (format: [x, y, w, h])
                prior_box = gt_data["gt_rect"][0]
                print("Loaded prior box from IR_label.json:", prior_box)
    else:
        print("IR_label.json not found. No prior knowledge available.")

    # This variable holds the last predicted bounding box. It is initialized using the ground truth if available.
    last_pred_box = prior_box.copy() if prior_box is not None else None

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load YOLOv12 model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier (if needed)
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = view_img and cv2.waitKey(1) == -1
        cudnn.benchmark = True  # speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=25.0)

    # Warm-up model
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # Preprocess image
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # scale to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply Non-Max Suppression (keeping only first image's detections)
        from ultralytics.utils.ops import non_max_suppression
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)[0]
        t2 = time_synchronized()

        # Optionally run classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections for current frame
        results = []
        # We assume one frame per iteration (or process each image in batch separately)
        if webcam:
            p, s, im0, frame = path[0], '', im0s[0].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        detections = []
        if pred is not None and pred.ndim:
            if pred.ndim == 1:  # single detection, reshape it
                pred = pred.view(1, -1)
            if len(pred):
                # Rescale boxes to original image size
                boxes = scale_coords(img.shape[2:], pred[:, :4], im0.shape).cpu().numpy()
                detections = pred.cpu().numpy()
                detections[:, :4] = boxes

        # Run tracker update if detections are available
        online_targets = []
        if len(detections):
            online_targets = tracker.update(detections, im0)

        # If we got a tracking result, pick the best (e.g. highest score) as the one-object hypothesis.
        if online_targets:
            best_target = max(online_targets, key=lambda t: t.score)
            tlwh = best_target.tlwh  # bounding box in [x, y, w, h]
            tlbr = best_target.tlbr  # bounding box in [x1, y1, x2, y2]
            tid = best_target.track_id
            last_pred_box = tlwh.copy()  # update the last predicted box
            # Draw the bounding box
            label = f'{tid}, {names[int(best_target.cls)]}' if not opt.hide_labels_name else f'{tid}'
            plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            # Save result string if needed
            results.append(f"{frame},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{best_target.score:.2f},-1,-1,-1\n")
        else:
            # No detection/tracking result: use last known bounding box (initially prior_box if available)
            if last_pred_box is not None:
                x, y, w, h = last_pred_box
                tlbr = [x, y, x + w, y + h]
                # Use a default ID (e.g. 0) and label
                label = 'pred'
                plot_one_box(tlbr, im0, label=label, color=colors[0], line_thickness=2)
                # Append dummy result with a default score of 1.0
                results.append(f"{frame},0,{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n")
            else:
                print(f"Frame {frame}: No detection and no prior bounding box available.")

        # Display or save results
        if view_img:
            cv2.imshow('BoT-SORT', im0)
            cv2.waitKey(1)  # 1 ms

        if save_img:
            p = Path(p)  # convert path to Path object
            save_path = str(save_dir / p.name)
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # video or stream
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:  # video file
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov12.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
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

    # CMC & ReID parameters
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid weights file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()
    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:
            for opt.weights in Path(opt.weights).expanduser().glob('*.pt'):
                strip_optimizer(opt.weights)
        detect()
