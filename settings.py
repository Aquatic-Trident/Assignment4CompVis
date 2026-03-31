from enum import Enum
import torch
import numpy

IMG_DIR = 'images'
ANNOTATION_DIR = 'annotations'

BATCH_SIZE = 4
DIR = 'data'
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.9

TARGET_SIZE = (112, 112)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
HYPER_PARAMETERS = (learning_rate)