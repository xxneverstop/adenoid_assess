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
from PIL import Image

from util.disk import getCache



kps_key = ['O', 'M', 'T', 'HP']
# CandidateInfoTuple(isNodule_bool=False, diameter_mm=0.0, series_uid='1....666426688999739595820', center_xyz=(25.31, -222.1, -158.99))
KPSInfoTuple = namedtuple('KPSInfoTuple', 'series_uid, kps_01, kps_02, kps_03, kps_04')

@functools.lru_cache(1)
def get_kps_list(requireOnDisk_bool=True):
    print(get_kps_list.__name__)
    jpg_list = glob.glob("data_300/images/*.jpg")
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in jpg_list}

    # print(len(presentOnDisk_set))

    kps_list = []
    with open('data_300/kps_csv_data/kps_data.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            kps_list.append(row)
            # print(row)
            # sys.exit()

    return kps_list

    

@functools.lru_cache(1, typed=True)
def read_image(image_path):
    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    return image_data


raw_cache = getCache('data_300/cache/', 'images_data_cache')
@raw_cache.memoize(typed=True)
def get_image_data(image_path):
    # print(get_image_data.__name__)
    return read_image(image_path)




class XRayDataset(Dataset):

    def __init__(self,
                 data_dir='data_300',
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
        # print('kps_list', self.kps_list)
        self.kps_list = sorted(self.kps_list, key=lambda x: x[0])

        series_id_list = [row[0] for row in self.kps_list]
        for filename in self.image_name_list:
            if filename[:-4] not in series_id_list:
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
        # print(np.array(self.kps_list[0][1:]).reshape(4, 2).astype(np.float32))

        # diameter_dict = {}
        # with open("data/part2/luna/annotations.csv", "r") as f:
        #     for row in list(csv.reader(f))[1:]:
        #         series_uid = row[0]
        #         annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
        #         annotationDiameter_mm = float(row[4])

        #         diameter_dict.setdefault(series_uid, []).append(
        #             (annotationCenter_xyz, annotationDiameter_mm),
        #         )

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
        bboxes_original = self._get_boxes_from_keypoints(keypoints_original)
        # print("bboxes_original", bboxes_original)
        # print("keypoints_original", keypoints_original)
        # sys.exit()
        bboxes_labels_original = ['air_way' for _ in bboxes_original]
        # sys.exit()
        if self.use_cache:
            image_original = get_image_data(image_path)
        else:
            image_original = cv2.imread(image_path)
            image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)


        if self.transform:
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [
                el[0:2] for kp in keypoints_original for el in kp
            ]
            # print("keypoints_original_flattened", keypoints_original_flattened)
            # Apply augmentations
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

        # print("bboxes_original", bboxes_original)
        # print("keypoints_original", keypoints_original)
        # print("bboxes", bboxes)
        # print("keypoints", keypoints)
        # sys.exit()

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

        # print("target", target)
        # print("target_original", target_original)

        if self.demo:
            return image, target, image_original, target_original
        else:
            return image, target

    def _get_kps_from_kps_info(self, kps_info):
        keypoints_original = np.array(kps_info[1:]).reshape(4, 2).astype(
            np.float32)
        # bboxes_original = self._get_boxes_from_keypoints(keypoints_original)
        keypoints_original = np.array(
            [np.insert(keypoints_original, 2, 1, axis=1).tolist()])

        # keypoints_original = keypoints_original.reshape(
        #     -1, 2, keypoints_original.shape[1])
        # print("2_keypoints_original", keypoints_original)

        # keypoints_original = np.stack(keypoints_original)
        # print("3_keypoints_original", keypoints_original)

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
        # boxes = keypoints[:, :, :2].reshape(-1, 4)
        # boxes = np.array(boxes, dtype=np.float32)

        # # Adjust coordinates where x2 < x1 or y2 < y1
        # adjusted_data = boxes.copy()
        # adjusted_data[:, [0, 2]] = np.sort(adjusted_data[:, [0, 2]], axis=1)
        # adjusted_data[:, [1, 3]] = np.sort(adjusted_data[:, [1, 3]], axis=1)

        # # Add margin of 10 to each box
        # margin = 10
        # for i in range(adjusted_data.shape[0]):  # Iterate over each box
        #     adjusted_data[i, 0] -= margin  # Subtract 10 from x1
        #     adjusted_data[i, 1] -= margin  # Subtract 10 from y1
        #     adjusted_data[i, 2] += margin  # Add 10 to x2
        #     adjusted_data[i, 3] += margin  # Add 10 to y2

        # margin = 10
        # # Add margin of 10 to each box
        # margin = 10
        # data = boxes.copy()
        # for i in range(data.shape[0]):  # Iterate over each box
        #     # Compare x-coordinates
        #     if data[i, 0] < data[i, 2]:  # If x1 < x2
        #         data[i, 0] -= margin  # Subtract 10 from x1
        #         data[i, 2] += margin  # Add 10 to x2
        #     else:
        #         data[i, 0] += margin  # Add 10 to x1
        #         data[i, 2] -= margin  # Subtract 10 from x2

        #     # Compare y-coordinates
        #     if data[i, 1] < data[i, 3]:  # If y1 < y2
        #         data[i, 1] -= margin  # Subtract 10 from y1
        #         data[i, 3] += margin  # Add 10 to y2
        #     else:
        #         data[i, 1] += margin  # Add 10 to y1
        #         data[i, 3] -= margin  # Subtract 10 from y2

        # return adjusted_data
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        flat_keypoints = keypoints.flatten()
        # Iterate through keypoints to find min and max x and y coordinates
        for i in range(0, len(flat_keypoints), 2):
            x = flat_keypoints[i]
            y = flat_keypoints[i + 1]
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        # Add a margin to the bounding box
        margin = 10  # Adjust this value to change the margin
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin

        # Construct bounding box coordinates
        bounding_box = [[min_x, min_y, max_x, max_y],
                        [min_x, min_y, max_x, max_y]]

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
