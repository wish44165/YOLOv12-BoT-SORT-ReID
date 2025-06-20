import os
import yaml
import shutil
import argparse
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# Download pretrained weight
target_dir = "weights"
os.makedirs(target_dir, exist_ok=True)
files_to_download = ["MOT_yolov12n.pt"]
for filename in files_to_download:
    downloaded_path = hf_hub_download(
        repo_id="wish44165/YOLOv12-BoT-SORT-ReID",
        filename=filename
    )
    shutil.copy(downloaded_path, os.path.join(target_dir, filename))
print(f"Downloaded files are saved to: {target_dir}")


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='./weights/MOT_yolov12n.pt', help='model name')
    parser.add_argument('--yaml_path', type=str, default='uav.yaml', help='The yaml path')
    parser.add_argument('--n_epoch', type=int, default=100, help='Total number of training epochs.')
    parser.add_argument('--n_patience', type=int, default=100, help='Early stopping patience.')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--single_cls', type=bool, default=True, help='Single class or not')
    parser.add_argument('--n_worker', type=int, default=8, help='Number of workers')
    parser.add_argument('--save_path', type=str, default='./runs/uav', help='Save path')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def make_temp_yaml_with_absolute_paths(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base = os.path.abspath(os.path.dirname(yaml_path))
    data['train'] = os.path.abspath(os.path.join(base, data['train']))
    data['val']   = os.path.abspath(os.path.join(base, data['val']))

    temp_path = '_temp_uav_abs.yaml'
    with open(temp_path, 'w') as f:
        yaml.safe_dump(data, f)

    return temp_path


if __name__ == '__main__':
    opt = parse_opt()
    
    # Patch the yaml to have absolute paths just for this session
    temp_yaml = make_temp_yaml_with_absolute_paths(opt.yaml_path)

    model = YOLO(opt.model_name)
    model.train(
        data=temp_yaml,
        epochs=opt.n_epoch,
        patience=opt.n_patience,
        batch=opt.bs,
        imgsz=opt.imgsz,
        device=0,
        workers=opt.n_worker,
        project=opt.save_path,
        single_cls=opt.single_cls
    )

    # Optional cleanup
    os.remove(temp_yaml)