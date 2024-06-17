# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import argparse
import sys
import os
import torch
from pathlib import Path
import csv

# Ensure YOLOv5 root is in the path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))  # Add the root to PATH for importing


from ultralytics.utils.plotting import Annotator, colors



from pathlib import Path
import cv2
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_boxes, increment_path)

from utils.torch_utils import select_device

def run(weights, source, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45,
        device='', save_csv=False, project='runs/detect', name='exp', line_thickness=1):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    # Setup save directory
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # Modified to False for auto-increment
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)
    (save_dir / "images").mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "detections.csv"

    # Load dataset
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # Scale images to [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

        # Process detections
        annotator = Annotator(im0s, line_width=line_thickness, example=str(names))
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
                    if save_csv:
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([names[int(cls)], f"{conf:.2f}"])
                    print(f"Building type is {names[int(cls)]} with confidence of {conf:.2f}")

        # Save annotated image
        save_path = save_dir / "images" / Path(path).name
        cv2.imwrite(str(save_path), annotator.result())
        print(f"Image saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, required=True, help="model.pt path(s)")
    parser.add_argument('--source', type=str, required=True, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--line-thickness', type=int, default=1, help='Bounding box line thickness')
    opt = parser.parse_args()

    run(opt.weights, opt.source, imgsz=opt.img_size, conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres, device=opt.device, save_csv=opt.save_csv,
        line_thickness=opt.line_thickness)


































#
# def run(weights, source, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45,
#         device='', save_csv=False, project='runs/detect', name='exp', line_thickness=1):
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=False)
#     stride, names = model.stride, model.names
#     imgsz = check_img_size(imgsz, s=stride)  # Check image size
#
#     # Setup save directory
#     save_dir = increment_path(Path(project) / name, exist_ok=True)
#     (save_dir / "labels").mkdir(parents=True, exist_ok=True)
#     (save_dir / "images").mkdir(parents=True, exist_ok=True)  # Directory to save images
#     csv_path = save_dir / "detections.csv"
#
#     # Load dataset
#     dataset = LoadImages(source, img_size=imgsz, stride=stride)
#     for path, img, im0s, vid_cap, s in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.float() / 255.0  # Scale images to [0, 1]
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         # Inference
#         pred = model(img, augment=False, visualize=False)
#         pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)
#
#         # Process detections
#         annotator = Annotator(im0s, line_width=line_thickness, example=str(names))
#         for det in pred:  # Detections per image
#             if len(det):
#                 det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
#                 for *xyxy, conf, cls in det:
#                     label = f"{names[int(cls)]} {conf:.2f}"
#                     annotator.box_label(xyxy, label, color=colors(int(cls), True))
#                     print(f"Building type is {names[int(cls)]} with confidence of {conf:.2f}")  # Formatted output
#                     if save_csv:
#                         with open(csv_path, 'a', newline='') as f:
#                             writer = csv.writer(f)
#                             writer.writerow([names[int(cls)], f"{conf:.2f}"])
#
#         # Save annotated image
#         save_path = str(save_dir / "images" / Path(path).name)
#         cv2.imwrite(save_path, annotator.result())
#
#         print(f"Image saved to {save_path}")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='model path')
#     parser.add_argument('--source', type=str, default='./data/images', help='source path for images')
#     parser.add_argument('--img-size', type=int, nargs='+', default=[640, 640], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
#     parser.add_argument('--line-thickness', type=int, default=1, help='Bounding box line thickness')
#     opt = parser.parse_args()
#
#     run(opt.weights, opt.source, imgsz=opt.img_size, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres,
#         device=opt.device, save_csv=opt.save_csv, line_thickness=opt.line_thickness)
#
