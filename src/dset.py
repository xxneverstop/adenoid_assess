import os
import sys
import glob
import csv
import json
import random
import copy
import functools
from collections import namedtuple

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import imgaug.augmenters as iaa
import imgaug

from util.disk import getCache

DATA_DIR = 'data_100_5kps'
IMAGE_GLOB_PATTERN = DATA_DIR + "/images/*.jpg"
KPS_CSV_PATH = DATA_DIR + '/kps_csv_data/kps_data.csv'
CACHE_ROOT = DATA_DIR + '/cache/'
CACHE_DIR = 'images_data_cache'

# KPS_KEY = ['O', 'M', 'T', 'HP']
KPS_KEY = ['Ba', 'Ar', 'T', 'PNS', 'W']


@functools.lru_cache(1)
def get_kps_list(requireOnDisk_bool=True):
    print(get_kps_list.__name__)
    jpg_list = glob.glob(IMAGE_GLOB_PATTERN)
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in jpg_list}

    # print(len(presentOnDisk_set))

    kps_list = []
    with open(KPS_CSV_PATH, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            kps_list.append(row)
            # print(row)
            # sys.exit()

    return kps_list

def get_croped_kps_data(image_path, keypoints):
    # assert keypoints.dtype == np.float32
    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    keypoints = keypoints.astype(np.float32)
    # print("keypoints.shape", keypoints.shape)

    # flat
    keypoints_flatted = [el[0:2] for kp in keypoints for el in kp]
    keypoints_flatted = np.array(keypoints_flatted).flatten()
    # print("keypoints_flatted.shape", keypoints_flatted.shape)

    # draw_points_from_labelme_with_data(image_data, keypoints_flatted.reshape(-1, 2))
    pairs = [(keypoints_flatted[i], keypoints_flatted[i + 1])
             for i in range(0, len(keypoints_flatted), 2)]
    # print(keypoints)

    Keypoint_list = []
    for pair in pairs:
        x, y = pair
        keypoint = imgaug.Keypoint(x=x, y=y)
        Keypoint_list.append(keypoint)


    # image_data = np.transpose(image_data, (2, 0, 1))

    keypoints = imgaug.KeypointsOnImage(Keypoint_list, shape=image_data.shape)
    seq = iaa.Sequential(
        [iaa.CropToFixedSize(width=1200, height=1500, position='center')])
    image_aug, kps_aug = seq(image=image_data, keypoints=keypoints)
    xy_array = kps_aug.to_xy_array()
    # print(xy_array)
    keypoints_aug = np.array([np.insert(xy_array, 2, 1, axis=1).tolist()]).astype(np.float32)
    # draw_points_from_labelme_with_data(image_aug, xy_array)
    return keypoints_aug


@functools.lru_cache(1, typed=True)
def get_croped_image_data(image_path):
    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    seq = iaa.Sequential(
        [iaa.CropToFixedSize(width=1200, height=1500, position='center')])
    image_aug = seq(image=image_data)

    return image_aug



@functools.lru_cache(1, typed=True)
def read_image(image_path):
    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    return image_data





raw_cache = getCache(CACHE_ROOT, CACHE_DIR)
@raw_cache.memoize(typed=True)
def get_image_data(image_path):
    # print(get_image_data.__name__)
    image = get_croped_image_data(image_path)
    return image




class XRayDataset(Dataset):

    def __init__(self,
                 data_dir=DATA_DIR,
                 transform=None,
                 demo=False,
                 augmentation_dict=None,
                 kps_list=None,
                 mode='train'):
        self.transform = transform
        self.demo = demo
        self.augmentation_dict = augmentation_dict
        if kps_list:
            self.kps_list = copy.copy(kps_list)
            self.use_cache = False
        else:
            self.kps_list = copy.copy(get_kps_list())
            self.use_cache = True

        self.demo = demo
        self.data_dir = data_dir
        self.transform = transform
        self.image_name_list = sorted(  
            os.listdir(os.path.join(data_dir, "images")))
        self.kps_list = sorted(self.kps_list, key=lambda x: x[0])
       
        series_id_list = [row[0] for row in self.kps_list]
        for filename in self.image_name_list:
            if filename[:-4] not in series_id_list:
                # print(filename[:-4])
                self.image_name_list.remove(filename)
        for filename in self.image_name_list:
            if filename[:-4] not in series_id_list:
                # print(filename[:-4])
                self.image_name_list.remove(filename)

        self.image_name_list = sorted(self.image_name_list)
        for i in range(len(self.kps_list)):
            if self.image_name_list[i][:-4] != self.kps_list[i][0]:
                print("data location not match", self.image_name_list[i],
                      self.kps_list[i][0])
                sys.exit()

        print('len(self.image_name_list)', len(self.image_name_list),
              'len(self.kps_list)', len(self.kps_list))
        assert len(self.image_name_list) == len(self.kps_list)

        # split the dataset
        if mode == 'train':  # 0.6 for train
            self.image_name_list = self.image_name_list[:int(
                0.6 * len(self.image_name_list))]
            self.kps_list = self.kps_list[:int(0.6 * len(self.kps_list))]
        elif mode == 'eval':  # 0.2 for validation
            self.image_name_list = self.image_name_list[
                int(0.6 *
                    len(self.image_name_list)):int(0.8 *
                                                   len(self.image_name_list))]
            self.kps_list = self.kps_list[int(0.6 * len(self.kps_list)
                                              ):int(0.8 * len(self.kps_list))]
        elif mode == 'test':  # 0.2 for test
            self.image_name_list = self.image_name_list[
                int(0.8 * len(self.image_name_list)):]
            self.kps_list = self.kps_list[int(0.8 * len(self.kps_list)):]
        else:
            self.image_name_list = self.image_name_list


    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, ndx):
        kps_info = self.kps_list[ndx]
        series_uid = kps_info[0]
        image_path = os.path.join(self.data_dir, "images",
                                  self.image_name_list[ndx])
        _, image_file = os.path.split(image_path)
        file_name, _ = os.path.splitext(image_file)

        # check series_uid == image file name
        assert kps_info[0] == file_name
        # convert kps

        keypoints_original = self._get_kps_from_kps_info(kps_info)

        # sys.exit()
        if self.use_cache:
            image_original = get_image_data(image_path)
            keypoints_original = get_croped_kps_data(image_path, keypoints_original)
        else:
            image_original = get_croped_image_data(image_path, keypoints_original)
            keypoints_original = get_croped_kps_data(image_path, keypoints_original)
        # print("keypoints_original:", keypoints_original)

        # print("----------------image_original:", image_original)


        bboxes_original = self._get_boxes_from_keypoints(keypoints_original)

        bboxes_labels_original = ['air_way' for _ in bboxes_original]

        if self.transform:
            keypoints_original_flattened = [
                el[0:2] for kp in keypoints_original for el in kp
            ]

            transformed = self.transform(
                image=image_original,
                bboxes=bboxes_original,
                bboxes_labels=bboxes_labels_original,
                keypoints=keypoints_original_flattened)
            image = transformed['image']
            bboxes = transformed['bboxes']

            points_array = np.array(transformed['keypoints'])
            points_array_with_visibility = np.hstack(
                (points_array, np.ones((len(points_array), 1))))

            # Reshape the points array to the desired format
            keypoints = points_array_with_visibility.reshape(
                (1, len(points_array), 3))

        else:
            image, bboxes, keypoints = image_original, bboxes_original, keypoints_original

        # Convert everything into a torch tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor(
            [1 for _ in bboxes],
            dtype=torch.int64)  # all objects are glue tubes
        target["image_id"] = torch.tensor([ndx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] -
                                                          bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        image = F.to_tensor(image)

        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor(
            [1 for _ in bboxes_original],
            dtype=torch.int64)  # all objects are glue tubes
        target_original["image_id"] = torch.tensor([ndx])
        target_original["area"] = (
            bboxes_original[:, 3] - bboxes_original[:, 1]) * (
                bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original),
                                                 dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original,
                                                       dtype=torch.float32)
        
        image_original = F.to_tensor(image_original)

        if self.demo:
            return image, target, image_original, target_original
        else:
            return image, target

    def _get_kps_from_kps_info(self, kps_info):
        keypoints_original = np.array(kps_info[1:]).reshape(len(KPS_KEY), 2)
        try:
            # Code that might raise an exception
            keypoints_original = keypoints_original.astype(np.float32)
        except ValueError:
            # Code to handle the exception
            print("ValueError", kps_info)
        # bboxes_original = self._get_boxes_from_keypoints(keypoints_original)
        keypoints_original = np.array(
            [np.insert(keypoints_original, 2, 1, axis=1).tolist()])

        return keypoints_original

    def _get_boxes_from_keypoints(self, keypoints):
        min_x = np.min(keypoints[..., 0])
        max_x = np.max(keypoints[..., 0])
        min_y = np.min(keypoints[..., 1])
        max_y = np.max(keypoints[..., 1])

        # Add a margin to the bounding box
        margin = 20  # Adjust the margin as needed
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin

        # Convert the coordinates to integers
        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(
            max_y)

        # Create the bounding box list
        bounding_box = np.array([[min_x, min_y, max_x, max_y]])

        return bounding_box


    def _find_max_num_coordinates(self):
        max_num_coordinates = 0
        for label_path in self.label_paths:
            with open(label_path, "r") as f:
                label_data = json.load(f)
                num_coordinates = len(label_data["shapes"][0]["points"])
                max_num_coordinates = max(max_num_coordinates, num_coordinates)
        return max_num_coordinates

    def _process_label(self, points):
        num_points = len(points)
        # print('max_num_coordinates:', self.max_num_coordinates)
        if num_points < self.max_num_coordinates:

            # Pad with zeros
            padded_points = points + [[0, 0]] * \
                (self.max_num_coordinates - num_points)
            label = np.array(padded_points, dtype=np.float32)
        else:
            # Truncate
            label = np.array(points[:self.max_num_coordinates],
                             dtype=np.float32)
        return label
