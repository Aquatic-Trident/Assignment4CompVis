import kagglehub
from torch.utils.data import Dataset, DataLoader
import os
import glob
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from settings import *
from ImgResize import *
import numpy as np

# Define transformations


def Download():
    # Download latest version
    path = kagglehub.dataset_download("andrewmvd/dog-and-cat-detection", output_dir=DIR)
    base_dir = DIR + '/'
    dataset = CatDogDataset(img_dir=base_dir+IMG_DIR, ann_dir=base_dir+ANNOTATION_DIR, transform=ComposeTransform())

    print("Path to dataset files:", path)
    return dataset

def Split(data, ratio):
    # Create Validation-set
    data1_size = int(len(data) * ratio)
    data2_size = int(len(data) - data1_size)
    data1, data2 = torch.utils.data.random_split(data, [data1_size, data2_size])

    return data1.dataset, data2.dataset

def Loader(data):
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


class CatDogDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        self.label_map = {"cat": 0, "dog": 1}  # Label mapping

    def parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        objects = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            label = self.label_map.get(name, -1)  # Default to -1 if unknown label
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

        return width, height, objects

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]

        image = Image.open(img_path).convert("RGB")
        _, _, objects = self.parse_annotation(ann_path)

        bboxes = []
        for obj in objects:
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = obj['bbox'][2]
            ymax = obj['bbox'][3]
            bboxes.append([xmin, ymin, xmax, ymax])  # in your assignment 4, you need to convert bbox into [x, y, w, h] and value range [0, 1]

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor([obj["label"] for obj in objects], dtype=torch.int64)

        if self.transform:
            for box_trans, params in self.transform[1]:
                bboxes = box_trans(image, bboxes, *params)
            image = self.transform[0](image)

        return image, bboxes, labels