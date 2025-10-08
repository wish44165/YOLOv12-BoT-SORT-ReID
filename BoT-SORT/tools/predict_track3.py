import argparse
import time
from pathlib import Path
import sys
from ultralytics import YOLO
import os
import cv2
import copy
import json
import torch
import torch.backends.cudnn as cudnn
from numpy import random

random.seed(44165)

sys.path.append('.')


# Inject local yolov12 path
local_ultra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov12'))
if local_ultra_path not in sys.path:
    sys.path.insert(0, local_ultra_path)

# Clean ultralytics-related modules
for mod in list(sys.modules):
    if mod.startswith("ultralytics"):
        del sys.modules[mod]

# Import block.py first
from ultralytics.nn.modules import block
sys.modules['ultralytics.nn.modules.block'] = block
print("✅ Using block.py (import successful)")

# Runtime fallback wrapper
def run_detect_with_fallback(detect_func, *args, **kwargs):
    try:
        # Try running detect normally
        detect_func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️ Runtime error with block.py ({e}), switching to block_v1.py...")
        from ultralytics.nn.modules import block_v1 as block
        sys.modules['ultralytics.nn.modules.block'] = block
        # Re-run detect with block_v1
        detect_func(*args, **kwargs)
        print("✅ Successfully ran detect() with block_v1.py")


from yolov12.models.experimental import attempt_load
from yolov12.utils.datasets import LoadStreams, LoadImages
from yolov12.utils.general import check_img_size, check_requirements, check_imshow, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov12.utils.plots import plot_one_box
from yolov12.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer


from huggingface_hub import hf_hub_download
import shutil

# Define your target directory
target_dir = "logs/sbs_S50"
os.makedirs(target_dir, exist_ok=True)  # Make sure the directory exists

# List of files to download
files_to_download = ["model_0016.pth", "config.yaml"]

# Download each file and move it to the target directory
for filename in files_to_download:
    downloaded_path = hf_hub_download(
        repo_id="wish44165/YOLOv12-BoT-SORT-ReID",
        filename=filename
    )
    shutil.copy(downloaded_path, os.path.join(target_dir, filename))

print(f"Downloaded files are saved to: {target_dir}")


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('save results to {}'.format(filename))

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))



    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load YOLOv12 model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # print(weights)
    # model = YOLO(weights[0])
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


    # first frame path
    labelPath = source + 'TestLabels_FirstFrameOnly/'

    # image folders path
    imagePath = source + 'Test_imgs/'

    folderList = os.listdir(imagePath)
    for foldern in folderList:
        folderPath = imagePath + foldern
        print('='*10, folderPath, '='*10)

        # Directories
        # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        save_dir = Path(Path(opt.project) / foldern, exist_ok=opt.exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(folderPath, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(folderPath, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

        # Create tracker
        tracker = BoTSORT(opt, frame_rate=30.0)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()


        # Prior Knowledge: Load ground truth from IR_label.json (expected format: {"gt_rect": [[x, y, w, h]]})
        # could be empty (MultiUAV-068)
        gt_path = labelPath + foldern + '.txt'
        prior_box = []

        # Define the original and new dimensions
        original_height, original_width, _ = cv2.imread(str(Path(Path(folderPath) / os.listdir(folderPath)[0]))).shape
        new_width = imgsz  # The width you resized to
        # Compute new height maintaining the aspect ratio, or set it explicitly if different
        new_height = int(original_height * (new_width / original_width))

        # Compute scaling factors
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        prior_box = []

        with open(gt_path, "r") as file:
            for line in file:
                values = line.strip().split(",")  # Split by comma
                obj_id = int(values[0])  # Extract ID
                # Extract bbox (x, y, width, height)
                x, y, w, h = map(float, values[2:6])
                
                # Apply scaling to convert coordinates
                x_scaled = x * scale_x
                y_scaled = y * scale_y
                w_scaled = w * scale_x
                h_scaled = h * scale_y
                
                # Calculate the new coordinates for top-left and bottom-right corners
                x1, y1 = x_scaled, y_scaled
                x2, y2 = x_scaled + w_scaled, y_scaled + h_scaled
                
                prior_box.append([x1, y1, x2, y2, 1., 0.])

        prior_box = torch.tensor(prior_box, device="cuda:0")
        last_pred_box = copy.deepcopy(prior_box)

        # First frame flag
        idx = 0

        # To record one box per frame
        os.makedirs(opt.save_path_answer, exist_ok=True)
        res_list = []

        for path, img, im0s, vid_cap in dataset:

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            # print('='*20)
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Run NMS to filter predictions
            from ultralytics.utils.ops import non_max_suppression
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)[0]  # Keep only the first image's detections
            # print(pred.shape)  # Expected shape: (N, 6)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            

            ################
            # first frame use gt, no need to detect
            ################
            if idx == 0:
                # init_loc = [[prior_box[0], prior_box[1], prior_box[0]+prior_box[2], prior_box[1]+prior_box[3], 1., 0.]]
                # init_loc = torch.tensor(init_loc, device="cuda:0")
                pred = prior_box
            else:
                pass

            idx += 1

            pred = [pred]    # [tensor([[], []])]
            # print(pred)

            # if prior is not empty
            if pred[0].numel() != 0:

                # Process detections
                results = []

                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    # Run tracker
                    detections = []

                    if det.ndim == 1:  # If det is 1D (single detection), reshape it to 2D
                        det = det.view(1, -1)  # Reshape to (1, 6) assuming [x1, y1, x2, y2, conf, class]
                    if len(det):
                        boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                        boxes = boxes.cpu().numpy()
                        detections = det.cpu().numpy()
                        detections[:, :4] = boxes

                    # print(detections)

                    online_targets, slosts_targets = tracker.update(detections, im0)

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    online_cls = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tlbr = t.tlbr
                        tid = t.track_id
                        tcls = t.cls
                        if tlwh[2] * tlwh[3] > opt.min_box_area:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_cls.append(t.cls)

                            # save results
                            results.append(
                                f"{i + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )

                            # frame ID, object ID, x1, y1, w, h, confidence=1, class=1, visibility ratio=1.0]
                            # print([idx, tid, round(tlwh[0], 2), round(tlwh[1], 2), round(tlwh[2], 2), round(tlwh[3], 2), 1, 1, 1])
                            res_list.append([idx, tid, round(tlwh[0], 2), round(tlwh[1], 2), round(tlwh[2], 2), round(tlwh[3], 2), 1, 1, 1])

                            if save_img or view_img:  # Add bbox to image
                                if opt.hide_labels_name:
                                    label = f'{tid}'
                                else:
                                    label = f'{tid}, {names[int(tcls)]}'
                                plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=1)
                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg

                    # Print time (inference + NMS)
                    # print(f'{s}Done. ({t2 - t1:.3f}s)')

                    # Stream results
                    if view_img:
                        cv2.imshow('BoT-SORT', im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)



            # debug #
            # if idx == 3:
            #     # save submit file
            #     answer_file = os.path.join(opt.save_path_answer, f"{foldern}.txt")
            #     with open(answer_file, "w") as file:
            #         for row in res_list:
            #             file.write(",".join(map(str, row)) + "\n")  # Convert each element to string and join with ','
            #     print(f'Done. ({time.time() - t0:.3f}s)')
            #     sys.exit()
                

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            # print(f"Results saved to {save_dir}{s}")


        ################
        # check number of box equals to the number of frame
        ################
        numberFrames = os.listdir(folderPath)
        # print('len(numberFrames) =', len(numberFrames))
        # print('len(res_list)', len(res_list))
        # save submit file
        answer_file = os.path.join(opt.save_path_answer, f"{foldern}.txt")
        with open(answer_file, "w") as file:
            for row in res_list:
                file.write(",".join(map(str, row)) + "\n")  # Convert each element to string and join with ','

        print('.jpg saved to: {}'.format(save_dir))
        print('.txt saved to: {}'.format(opt.save_path_answer))
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

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    # Additional arguments
    parser.add_argument('--save_path_answer', type=str, default=None, help='Path to save the label files. If not set, "_label" is appended to source.')

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in Path(opt.weights).expanduser().glob('*.pt'):
                strip_optimizer(opt.weights)
        run_detect_with_fallback(detect)
