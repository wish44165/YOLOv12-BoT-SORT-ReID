import argparse
from ultralytics import YOLO


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='yolov12n.pt', help='model name')
    parser.add_argument('--yaml_path', type=str, default='uav.yaml', help='The yaml path')
    parser.add_argument('--n_epoch', type=int, default=100, help='Total number of training epochs.')
    parser.add_argument('--n_patience', type=int, default=100, help='Number of epochs to wait without improvement in validation metrics before early stopping the training.')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--single_cls', type=bool, default=True, help='single class or not')
    parser.add_argument('--n_worker', type=int, default=8, help='Number of workers')
    parser.add_argument('--save_path', type=str, default='./runs/uav', help='Save path')
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    model = YOLO(opt.model_name)

    model.train(
        data=opt.yaml_path,
        epochs=opt.n_epoch,
        patience=opt.n_patience,
        batch=opt.bs,
        imgsz=opt.imgsz,
        device=0,
        workers=opt.n_worker,
        project=opt.save_path,
        single_cls=opt.single_cls
    )