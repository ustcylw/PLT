import os, sys
import torch


class CONFIGS:
    PRE_SYMBOL = 'yolov1'
    POST_SYMBOL = '000001'

    MODE = 'test'
    SEED = 7
    

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(ROOT_DIR):
        raise ValueError(f'PROJECT NOT EXIST!!! {ROOT_DIR}')
    DATASET_DIR = '/data/ylw/datasets/voc/VOC2007'
    CHECKPOINT_DIR = os.path.join(os.path.dirname(ROOT_DIR), f'checkpoints/{PRE_SYMBOL}{POST_SYMBOL}')
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    LOGS_DIR = os.path.join(ROOT_DIR, f'logs/{PRE_SYMBOL}_{POST_SYMBOL}')

    INPUT_SHAPE = (418, 418, 3)

    PRETRAINED_MODEL_NAME = 'epoch=79-val_loss=3.98-other_metric=0.00.ckpt'
    PRETRAINED = os.path.join(ROOT_DIR, f'checkpoints/{PRETRAINED_MODEL_NAME}')

    AVAIL_GPUS = min(1, torch.cuda.device_count())
    BATCH_SIZE = 1 # 32  # 256 if AVAIL_GPUS else 64
    NUM_WORKERS = 4  # int(os.cpu_count() / 2)



    LEARNING_RATE = 1e-3

    TEST_SAVE = True
    TEST_SAVE_DIR = os.path.join(ROOT_DIR, 'results')
    TEST_SHOW = True

    VOC_CLASSES = (    # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

    Color = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]
    ]
