import torch
import torch.nn as nn
import cv2
import numpy as np
import time
from torchvision import transforms
from PIL import Image
from collections import Counter
import os

# Konstanta dan Hyperparameter
IMAGE_SIZE = 416
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 6
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

ANCHORS = [
    [(0.3400, 0.8897), (0.1355, 0.7105), (0.2592, 0.4301)],
    [(0.2951, 0.7624), (0.1102, 0.1949), (0.2937, 0.5041)],
    [(0.1004, 0.3905), (0.2266, 0.8294), (0.1941, 0.4585)]
]

CUSTOM_CLASSES = [
    "UltraMilk_Layak",
    "UltraMilk_Rusak",
    "FrisianFlag_Layak",
    "FrisianFlag_Rusak",
    "TehKotak_Layak",
    "TehKotak_Rusak"
]

CONF_THRESHOLD = 0.85
NMS_IOU_THRESH = 0.3

# Warna Gelap yang Berbeda (BGR format untuk OpenCV)
COLORS = [
    (0, 0, 128),     # Navy (Dark Blue)
    (128, 0, 0),     # Maroon (Dark Red)
    (0, 100, 0),     # Dark Green
    (128, 128, 0),   # Olive
    (75, 0, 130),    # Indigo
    (139, 69, 19),   # Saddle Brown
]

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
])

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x))) if self.use_bn_act else self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5)*3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (self.pred(x)
                .reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2))

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections.pop()], dim=1)
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNNBlock(in_channels, out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=1 if kernel_size == 3 else 0))
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        return layers

def iou_width_height(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0]*boxes1[..., 1] + boxes2[..., 0]*boxes2[..., 1] - intersection
    )
    return intersection / (union + 1e-6)

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3]/2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4]/2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3]/2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4]/2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3]/2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4]/2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3]/2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4]/2
    elif box_format == "corners":
        box1_x1 = boxes_preds[...,0:1]
        box1_y1 = boxes_preds[...,1:2]
        box1_x2 = boxes_preds[...,2:3]
        box1_y2 = boxes_preds[...,3:4]
        box2_x1 = boxes_labels[...,0:1]
        box2_y1 = boxes_labels[...,1:2]
        box2_x2 = boxes_labels[...,2:3]
        box2_y2 = boxes_labels[...,3:4]
    else:
        raise ValueError("Invalid box_format.")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0)*(y2-y1).clamp(0)
    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))
    return intersection/(box1_area+box2_area-intersection+1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    assert isinstance(bboxes, list)
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    BATCH_SIZE = predictions.shape[0]
    num_anchors = anchors.shape[0]
    box_predictions = predictions[...,1:5]
    if is_preds:
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)
        box_predictions[...,0:2] = torch.sigmoid(box_predictions[...,0:2])
        box_predictions[...,2:] = torch.exp(box_predictions[...,2:])*anchors
        scores = torch.sigmoid(predictions[...,0:1])
        best_class = torch.argmax(predictions[...,5:], dim=-1, keepdim=True)
    else:
        scores = predictions[...,0:1]
        best_class = predictions[...,5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(BATCH_SIZE, num_anchors, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = (box_predictions[...,0:1] + cell_indices) / S
    y = (box_predictions[...,1:2] + cell_indices.permute(0, 1, 3, 2, 4)) / S
    w_h = box_predictions[...,2:] / S
    converted_bboxes = torch.cat((best_class.float(), scores, x, y, w_h), dim=-1)
    return converted_bboxes.reshape(BATCH_SIZE, num_anchors*S*S, 6)

model = YOLOv3(num_classes=NUM_CLASSES).to(DEVICE)
checkpoint = torch.load('/home/individual1/Documents/checkpoint_model4.pth.tar', map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

scaled_anchors = (torch.tensor(ANCHORS)*torch.tensor(S).unsqueeze(1).unsqueeze(2)).to(DEVICE)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat mengakses kamera.")
    exit()

# Atur ukuran input kamera agar lebih ringan
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_number = 0
total_time = 0
print("Deteksi real-time dimulai. Tekan 'q' untuk keluar.")

window_name = "Deteksi Real-Time"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Agar dapat di-maximize & minimize
cv2.resizeWindow(window_name, 800, 600)          # Ukuran awal window

font_scale = 0.3
font_thickness = 1
box_thickness = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    start_time = time.time()

    # Preprocessing
    frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img = test_transforms(img_pil).to(DEVICE).unsqueeze(0)

    # Inference
    with torch.no_grad():
        predictions = model(img)

    # Post-processing
    bboxes = []
    for i in range(3):
        S_current = predictions[i].shape[2]
        anchor = scaled_anchors[i]
        boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S_current, is_preds=True)
        bboxes += boxes_scale_i[0].cpu().numpy().tolist()

    nms_boxes = non_max_suppression(
        bboxes,
        iou_threshold=NMS_IOU_THRESH,
        threshold=CONF_THRESHOLD,
        box_format="midpoint",
    )

    # Gambar bounding box dan teks
    for box in nms_boxes:
        class_pred = int(box[0])
        confidence = box[1]
        x, y, w, h = box[2], box[3], box[4], box[5]

        x = x * frame.shape[1]
        y = y * frame.shape[0]
        w = w * frame.shape[1]
        h = h * frame.shape[0]

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[class_pred], box_thickness)
        label = f"{CUSTOM_CLASSES[class_pred]} {confidence:.2f}"
        # Tampilkan teks
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLORS[class_pred], font_thickness)

    end_time = time.time()
    inference_time = end_time - start_time
    total_time += inference_time
    frame_fps = 1 / inference_time if inference_time > 0 else 0

    fps_text = f"FPS: {frame_fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

    # Tampilkan hasil pada layar
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_number += 1

cap.release()
cv2.destroyAllWindows()

average_fps = frame_number / total_time if total_time > 0 else 0
print(f"Rata-rata FPS: {average_fps:.2f}")
