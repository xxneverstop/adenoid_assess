import glob
import os
import json
import time

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

import matplotlib.pyplot as plt

import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from src.utils import delete_test_file
from src.utils import handle_raw_data_dir
from src.utils import draw_points_from_labelme
from src.utils import draw_points_from_labelme_with_data
from src.utils import rename_label_files
from src.utils import label_to_csv
from src.utils import visualize_prediction_comparision
from .dset import XRayDataset
from .utils import handle_raw_data_dir
from util.util import collate_fn
from .dset import KPS_KEY


def cache_test():
    print(cache_test.__name__)

    dset = XRayDataset('data')

    dloader = DataLoader(dset, batch_size=4, shuffle=True)

    start_time = time.time()

    for batch_ndx, images in enumerate(dloader):
        # if batch_ndx > len(dloader) / 2:
        #     break
        # plt.imshow(images[0])
        # plt.axis('off')  # Turn off axis
        # plt.show()
        pass

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print("load images cost(no cache):", elapsed_time, "seconds")
    print("load images cost(with cache):", elapsed_time, "seconds")



mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def train_transform():
    return A.Compose([
        A.Sequential([
            A.SafeRotate(limit=30, p=1), # Random rotation of an image by 90 degrees zero or more times
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.HorizontalFlip(p=0.5),  # Randomly flip the image horizontally
            A.VerticalFlip(p=0.5),  # Randomly flip the image vertically
            # A.Normalize(mean=mean, std=std),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=50, shift_limit_x=0.2, shift_limit_y=0.2, p=0.5),  # Randomly shift, scale, and rotate the image
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0)
        ], p=1)

    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )



def visualize_augmentation(image, bboxes_augmented, keypoints_augmented, image_original, bboxes_original, keypoints_original):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the original image with keypoints
    axes[0].imshow(image_original)
    # axes[0].scatter(keypoints_original[:, 0], keypoints_original[:, 1], c='r', marker='o')
    axes[0].scatter(keypoints_original[:, 0], keypoints_original[:, 1], s=4, c='red', marker='o')
    for i, label in enumerate(KPS_KEY):
        axes[0].annotate(label, (keypoints_original[i][0], keypoints_original[i][1]), textcoords="offset points", xytext=(0,10), ha='center')

    # for box in bboxes_original:
    #     x, y, w, h = box
    #     rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
    #     axes[0].add_patch(rect)

    axes[0].set_title('Original Image with Keypoints')


    # Plot the augmented image with keypoints
    axes[1].imshow(image)
    # axes[1].scatter(keypoints_augmented[:, 0], keypoints_augmented[:, 1], c='r', marker='o')
    axes[1].scatter(keypoints_augmented[:, 0], keypoints_augmented[:, 1], s=4, c='red', marker='o')
    for i, label in enumerate(KPS_KEY):
        axes[1].annotate(label, (keypoints_augmented[i][0], keypoints_augmented[i][1]), textcoords="offset points", xytext=(0,10), ha='center')

    # for box in bboxes_augmented:
    #     x, y, w, h = box
    #     rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
    #     axes[1].add_patch(rect)
    axes[1].set_title('Augmented Image with Keypoints')


    # Show the plot
    plt.show()




import sys

def denormalize_image_tensor(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(2)
    denormalized_image = (image_tensor * std + mean) * max_pixel_value
    return denormalized_image.byte()


def visualize_test(data_loader):

    for i in range(4):
        batch = next(iter(data_loader))
        # image, item_info_dict, image_original, item_info_dict_original
        # print("batch[1][0]['boxes']", batch[1][0]['boxes'])
        # print("batch[1][0]['keypoints']", batch[1][0]['keypoints'])
        bboex_transformed = batch[1][0]['boxes'].numpy()
        keypoint_transformed = batch[1][0]['keypoints'][..., :2].numpy()[0]
        # print("bboex_transformed", bboex_transformed)
        # print("keypoint_transformed", keypoint_transformed)

        # print("batch[3][0]['boxes']", batch[3][0]['boxes'])
        # print("batch[3][0]['keypoints']", batch[3][0]['keypoints'])
        bboex_original = batch[3][0]['boxes'].numpy()
        keypoint_original = batch[3][0]['keypoints'][..., :2].numpy()[0]
        # print("bboex_original", bboex_original)
        # print("keypoint_original", keypoint_original)


        # denormalize
        # print(batch[0][0])
        # image_data = denormalize_image_tensor(batch[0][0])
        # image = (image_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # print(image_data.shape)
        # image = (image_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)



        image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


        image_original = (batch[2][0].permute(1, 2, 0).numpy() * 255).astype(
            np.uint8)

        print(image_original.shape)
        visualize_augmentation(image, bboex_transformed, keypoint_transformed, image_original,
                               bboex_original, keypoint_original)

def augmentation_test():
    print(augmentation_test.__name__)

    dset = XRayDataset(transform=train_transform(), demo=True)

    dloader = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    start_time = time.time()

    visualize_test(dloader)


from .utils import delete_json_files
from .utils import move_images_to_dir


def dset_test():
    print(dset_test.__name__)


    dset = XRayDataset(transform=train_transform(), demo=True)

    dloader = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    iterator = iter(dloader)
    batch = next(iterator)
    # print("Original targets:\n", batch[3], "\n\n")
    # print("Transformed targets:\n", batch[1])


def process_raw_file(raw_data_dir):
    print("process_raw_file")
    handle_raw_data_dir(raw_data_dir)

    label_to_csv(raw_data_dir, raw_data_dir + '/kps_csv_data')

    delete_json_files(raw_data_dir)

    move_images_to_dir(raw_data_dir, raw_data_dir + '/images')

from .dset import DATA_DIR
if __name__ == "__main__":
    print(__name__)
    
    raw_data_dir = 'data_100_5kps'   
    process_raw_file(raw_data_dir)
    

    # augmentation_test()
