import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from typing import Tuple


VOC_COLORS = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128], [224, 224, 192]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor', 'edge']

COLORMAP = dict(zip([tuple(lst) for lst in VOC_COLORS], range(22)))
COLORMAP[(224, 224, 192)] = 0

TRAIN_PATH = "ImageSets/Segmentation/train.txt"
TEST_PATH = "ImageSets/Segmentation/val.txt"
IMAGE_PATH = "JPEGImages"
SEGMENTATION_LABEL_PATH = "SegmentationClass"

def label_preprocessing(label: torch.Tensor):
    int_label = torch.zeros(label.shape[1:])
    for x, row in enumerate(label.permute(1, 2, 0)):
        for y, color in enumerate(row):
            int_label[x, y] = COLORMAP[tuple(color.tolist())]
    
    return int_label

def random_crop(img: torch.Tensor, label: torch.Tensor, size: Tuple[int, int]=(320, 480)):
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(img, size)
    cropped_img = torchvision.transforms.functional.crop(img, i, j, h, w)
    cropped_label = torchvision.transforms.functional.crop(label, i, j, h, w)

    return cropped_img, cropped_label

class VocDataset(Dataset):
    def __init__(self, dataset_path: str, train: bool=True, shape: Tuple[int, int]=(320, 480)):
        if train:
            with open(os.path.join(dataset_path, TRAIN_PATH)) as f:
                file_list = [string for string in f.read().split('\n') if string]
        else:
            with open(os.path.join(dataset_path, TEST_PATH)) as f:
                file_list = [string for string in f.read().split('\n') if string]

        self.image_list = []
        self.label_list = []
        self.shape = shape

        for file in file_list:
            img = torchvision.io.read_image(
                os.path.join(dataset_path, IMAGE_PATH, file + '.jpg'),
                mode=torchvision.io.image.ImageReadMode.RGB)
            if img.shape[1] < shape[0] or img.shape[2] < shape[1]:
                continue
            self.image_list.append(img)
            label = torchvision.io.read_image(
                os.path.join(dataset_path, SEGMENTATION_LABEL_PATH, file + '.png'),
                mode=torchvision.io.image.ImageReadMode.RGB)
            self.label_list.append(label)
        
        self.size = len(self.image_list)
        print("Read {} images.".format(self.size))

    def __getitem__(self, idx: int):
        cropped_img, cropped_label = random_crop(
            self.image_list[idx], self.label_list[idx], self.shape
        )
        # int_label = cropped_label
        int_label = label_preprocessing(cropped_label)
        return cropped_img, int_label
    
    def __len__(self):
        return self.size
