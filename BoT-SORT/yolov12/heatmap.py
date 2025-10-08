# --------------------------------------------------------
# Based on yolov12
# https://github.com/sunsmarterjie/yolov12/issues/74
# --------------------------------------------------------'

import os
import cv2
import torch
import types
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms


imgs_dir_8x = './imgs_dir_8x'
os.makedirs(imgs_dir_8x, exist_ok=True)
up_size = 8

for img in os.listdir('imgs_dir'):
    image = Image.open(os.path.join('imgs_dir', img))
    new_size = (image.width * up_size, image.height * up_size)
    upscaled_image = image.resize(new_size, Image.BILINEAR)
    upscaled_image.save(os.path.join(imgs_dir_8x, img))

def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], [] 
        tmp = 0
        for m in self.model:
            if m.f != -1:  
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f] 
            if profile:
                self._profile_one_layer(m, x, dt)
            
            tmp += 1
            if tmp ==7:  # you can decide the output layer here
                return x

            x = m(x) 
            y.append(x if m.i in self.save else None) 
        return x

def heatmap(model, img_path, save_file='./outputs'):
    save_img_path = os.path.join(save_file, img_path.split('/')[-1].replace('.jpg', '_heatmap.jpg'))
    
    # Load image
    img = Image.open(img_path)
    w, h = img.size

    # Transform image for model
    transform = transforms.Compose([
        transforms.Resize((h // 3 * 2 // 32 * 32, w // 3 * 2 // 32 * 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = transform(img).unsqueeze(0)
    feature = model.model._predict_once(img)

    # Process output
    outputs = feature.squeeze(0).mean(dim=0)  # Average across channels
    heatmap = outputs.detach().cpu().numpy()

    # Normalize heatmap to [0, 255]
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)

    # Resize back to original image size (640, 512)
    heatmap_resized = cv2.resize(heatmap_color, (640, 512), interpolation=cv2.INTER_CUBIC)

    # Save using OpenCV (no white borders)
    cv2.imwrite(save_img_path, heatmap_resized)


if __name__ == "__main__":
    model = YOLO('./weights/v1/MOT_yolov12n.pt')
    setattr(model.model, "_predict_once", types.MethodType(_predict_once, model.model))

    heatmap_dir = './outputs'
    os.makedirs(heatmap_dir, exist_ok=True)

    for path in os.listdir('./imgs_dir_8x'):
        img_path = os.path.join('./imgs_dir_8x', path)
        heatmap(model, img_path, save_file=heatmap_dir)
