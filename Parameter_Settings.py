import os
import cv2
import torch
from torchvision import transforms

DATASET = 'CUSTOM_DATSET'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1
BATCH_SIZE = 32  # Sesuaikan dengan memori GPU Anda
IMAGE_SIZE = 416
NUM_CLASSES = 6
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 2
CONF_THRESHOLD = 0.5
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
CHECKPOINT_FILE = "/content/drive/MyDrive/Skripsi/checkpoint_model.pth.tar"
PIN_MEMORY = True
LOAD_MODEL = os.path.exists(CHECKPOINT_FILE)
SAVE_MODEL = True

IMG_TRAIN_DIR = image_train_dir
LABEL_TRAIN_DIR = label_train_dir
IMG_TEST_DIR = image_test_dir
LABEL_TEST_DIR = label_test_dir

ANCHORS = [
    [(0.3400, 0.8897), (0.1355, 0.7105), (0.2592, 0.4301)],
    [(0.2951, 0.7624), (0.1102, 0.1949), (0.2937, 0.5041)],
    [(0.1004, 0.3905), (0.2266, 0.8294), (0.1941, 0.4585)]
]

# Transformasi untuk pelatihan
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Pastikan ukuran gambar konsisten
    transforms.RandomAffine(degrees=20, translate=None, scale=None, shear=15),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
])

# Transformasi untuk pengujian
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Pastikan ukuran gambar konsisten
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
])

# Buat DataLoader untuk pelatihan dan pengujian
train_dataset = YOLODataset(IMG_TRAIN_DIR, LABEL_TRAIN_DIR, ANCHORS, IMAGE_SIZE, S, NUM_CLASSES, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

test_dataset = YOLODataset(IMG_TEST_DIR, LABEL_TEST_DIR, ANCHORS, IMAGE_SIZE, S, NUM_CLASSES, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

CUSTOM_CLASSES = [
    "UltraMilk_Layak",
    "UltraMilk_Rusak",
    "FrisianFlag_Layak",
    "FrisianFlag_Rusak",
    "TehKotak_Layak",
    "TehKotak_Rusak"
]

