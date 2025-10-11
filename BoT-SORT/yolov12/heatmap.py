# --------------------------------------------------------
# Based on yolov12
# https://github.com/sunsmarterjie/yolov12/issues/74
# --------------------------------------------------------'

import os
import cv2
import torch
import types
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

# ------------------------- CONFIG -------------------------
IMG_SRC_DIR = './imgs_dir'       # original images (will be upscaled)
IMG_UP_DIR = './imgs_dir_8x'     # upscaled images (script will create)
OUTPUT_DIR = './outputs'         # saved heatmaps
WEIGHTS_PATH = './weights/v1/MOT_yolov12n.pt'
UPSCALE_FACTOR = 8
SAVE_SIZE = (640, 512)           # final saved heatmap size (width, height)
# ----------------------------------------------------------

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# create folders if missing
os.makedirs(IMG_SRC_DIR, exist_ok=True)
os.makedirs(IMG_UP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def upscale_images(src_dir: str, dst_dir: str, up_size: int = 8):
    """Upscale images in src_dir by up_size and save to dst_dir."""
    imgs = [f for f in sorted(os.listdir(src_dir)) if f.lower().endswith(IMAGE_EXTS)]
    if not imgs:
        print(f"[WARN] No images found in {src_dir}. Place images there before running.")
        return 0

    for fname in imgs:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        try:
            with Image.open(src_path) as im:
                im = im.convert('RGB')
                new_size = (im.width * up_size, im.height * up_size)
                up_im = im.resize(new_size, Image.BILINEAR)
                up_im.save(dst_path)
                print(f"[INFO] Upscaled {src_path} -> {dst_path} ({new_size})")
        except Exception as e:
            print(f"[ERROR] Upscaling {src_path}: {e}")
    return len(imgs)


def _predict_once(self, x, profile=False, visualize=False, embed=None):
    """
    Minimal replacement of model._predict_once to return intermediate x at a selected layer.
    The user used tmp==7 as the layer index — keep that behavior.
    """
    y, dt = [], []
    tmp = 0
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        if profile:
            # If model has _profile_one_layer, call it; otherwise ignore
            try:
                self._profile_one_layer(m, x, dt)
            except Exception:
                pass

        tmp += 1
        # return the activation BEFORE applying layer 7's op (as original code desired)
        if tmp == 7:
            return x

        x = m(x)
        y.append(x if m.i in self.save else None)
    return x


def heatmap(model: YOLO, img_path: str, save_file: str = OUTPUT_DIR):
    """Generate and save a heatmap for a single image path."""
    try:
        os.makedirs(save_file, exist_ok=True)
        base = os.path.basename(img_path)
        save_img_name = base.rsplit('.', 1)[0] + '_heatmap.jpg'
        save_img_path = os.path.join(save_file, save_img_name)

        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size

        # Compute target size - similar logic to the user's working version.
        tgt_h = ((orig_h * 2) // 3) // 32 * 32
        tgt_w = ((orig_w * 2) // 3) // 32 * 32
        # If the math above produced 0, force minimum 32
        tgt_h = max(32, int(tgt_h or 32))
        tgt_w = max(32, int(tgt_w or 32))

        # Note: torchvision.transforms.Resize expects (height, width)
        transform = transforms.Compose([
            transforms.Resize((tgt_h, tgt_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_t = transform(pil_img).unsqueeze(0)  # shape (1, C, H, W)

        # Move to model device
        try:
            device = next(model.model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        img_t = img_t.to(device)

        model.model.eval()

        # No grad to avoid extra memory
        with torch.no_grad():
            feature = model.model._predict_once(img_t)

        # If model returned a list/tuple, pick the last tensor found (robust)
        if isinstance(feature, (list, tuple)):
            found = None
            for elem in reversed(feature):
                if isinstance(elem, torch.Tensor):
                    found = elem
                    break
            if found is None:
                raise ValueError("No tensor found in returned feature list/tuple.")
            feature = found

        if not isinstance(feature, torch.Tensor):
            raise ValueError(f"Returned feature is not a tensor: {type(feature)}")

        # produce heatmap by averaging channels
        if feature.dim() == 4:
            # (B, C, H, W) -> remove batch and mean channels
            outputs = feature.squeeze(0).mean(dim=0)
        elif feature.dim() == 3:
            # (C, H, W) -> mean channels
            outputs = feature.mean(dim=0)
        else:
            raise ValueError(f"Unsupported feature.dim() == {feature.dim()}")

        heatmap = outputs.detach().cpu().numpy()
        # Normalize to 0..255 uint8
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)

        # Resize back to desired save size (width, height)
        heatmap_resized = cv2.resize(heatmap_color, SAVE_SIZE, interpolation=cv2.INTER_CUBIC)

        # Save
        ok = cv2.imwrite(save_img_path, heatmap_resized)
        print(f"[INFO] wrote {save_img_path}: {ok}")

    except Exception as e:
        print(f"[ERROR] processing {img_path}: {e}", flush=True)


def main():
    # 1) Upscale images from IMG_SRC_DIR -> IMG_UP_DIR
    upcount = upscale_images(IMG_SRC_DIR, IMG_UP_DIR, up_size=UPSCALE_FACTOR)
    if upcount == 0:
        print("[ABORT] No source images to process. Put files in", IMG_SRC_DIR)
        return

    # 2) Load model and bind predict hook
    print("[INFO] Loading YOLO model:", WEIGHTS_PATH)
    model = YOLO(WEIGHTS_PATH)
    # attach our _predict_once to model.model (so code calls model.model._predict_once)
    setattr(model.model, "_predict_once", types.MethodType(_predict_once, model.model))

    # If CUDA available, the ULTRALYTICS module may have auto device placement; we just ensure input sent to same device.

    # 3) Iterate upscaled images and produce heatmaps
    up_imgs = [f for f in sorted(os.listdir(IMG_UP_DIR)) if f.lower().endswith(IMAGE_EXTS)]
    if not up_imgs:
        print(f"[WARN] No images found in {IMG_UP_DIR} after upscaling.")
        return

    for fname in up_imgs:
        img_path = os.path.join(IMG_UP_DIR, fname)
        print(f"[INFO] processing {img_path}")
        heatmap(model, img_path, save_file=OUTPUT_DIR)


if __name__ == "__main__":
    main()
