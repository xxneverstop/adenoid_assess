import glob
import os
import json
import time

import numpy as np
import cv2
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import to_tensor

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




def test1():
    print("test1")
    image_path = os.path.join("dataset", "image", "00001.jpg")
    label_path = os.path.join("dataset", "label", "00001.json")
    draw_points_from_labelme(image_path, label_path)


def affine_test():
    # Load and preprocess an example image
    image = plt.imread("dataset/image/00001.jpg")  # Load image
    json_file_path = "dataset/label/00001.json"
    with open(json_file_path, "r") as file:
        label_data = json.load(file)
    points = label_data["shapes"][0]["points"]
    label = np.array(points, dtype=np.float32)
    print(type(label), type(image))
    print(label.shape, image.shape)
    draw_points_from_labelme_with_data(image, label)
    image = (
        torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    )  # Convert to tensor and add batch dimension
    image /= 255.0  # Normalize to [0, 1]

    # # Generate an affine transformation matrix for scaling by 2 along the x-axis and y-axis
    # theta = torch.tensor([[[0.5, 0, 0], [0, 0.5, 0]]], dtype=torch.float)

    # Generate an affine transformation matrix for shearing along the x-axis
    theta_flip = torch.tensor([[[-1, 0, 0], [0, 1, 0]]], dtype=torch.float)
    # theta = torch.eye(4)
    # for i in range(3):
    #     theta[i,i] *= -1

    # # Generate affine transformation matrix for translating by (10, 20) pixels
    # theta = torch.tensor([[[1, 0, 0.5], [0, 1, 0.5]]], dtype=torch.float)

    # Generate grid of coordinates
    grid = F.affine_grid(theta_flip, image.size())

    # Perform affine transformation using grid_sample
    rotated_image = F.grid_sample(image, grid)

    # Convert the tensor back to an image and display
    rotated_image = rotated_image.squeeze(0).permute(1, 2, 0).numpy()

    # Convert the original tensor back to an image and display
    original_image = image.squeeze(0).permute(1, 2, 0).numpy()

    # draw_points_from_labelme_with_data(rotated_image, label)
    label_augmented_flip = torch.matmul(torch.tensor(label), theta_flip[:, :2, :2].transpose(1, 2)) + theta_flip[:, :, 2].unsqueeze(1)
    draw_points_from_labelme_with_data(rotated_image, label_augmented_flip.squeeze(0).numpy())
    print('label_augmented_flip:', label_augmented_flip.shape)

    print(label.shape)
    print(image.shape)
    draw_points_from_labelme_with_data(image.squeeze(0).permute(1, 2, 0).cpu().numpy(), label)


    # Display the images
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(rotated_image)
    plt.title("Rotated Image")
    plt.axis("off")

    plt.show()


def show_4_images_one_row(images):

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        # [channels, height, width] to [height, width, channels]
        images_resized = resize(images[i], (256, 256))
        image_np = images_resized.permute(1, 2, 0).cpu().numpy()
        # Display the image using matplotlib
        axes[i].imshow(image_np)
        axes[i].axis("off")
    plt.show()


def test_image_after_augmentation_with_transforms():
    data_dir = "dataset"
    original_ds = XRayDataset(data_dir=data_dir)
    print(len(original_ds))

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
        ]
    )

    train_transform1 = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
        ]
    )

    raw_ds = XRayDataset(
        data_dir=data_dir, transform=transforms.Compose([transforms.ToTensor()])
    )
    print(type(raw_ds))
    print(len(raw_ds))

    raw_list4 = []
    for i in range(4):
        raw_image, _ = raw_ds[i]
        raw_list4.append(raw_image)
    print(type(raw_list4))
    # show
    # show_4_images_one_row(raw_list4)

    aug_list4 = []
    for i in range(4):
        img, _ = original_ds[i]
        aug_image = train_transform(img)
        aug_list4.append(aug_image)
    # show_4_images_one_row(aug_list4)

    # label does not match after augmentation
    image, label = raw_ds[0]
    print(type(label))
    print(label.shape)
    print(image.shape)
    draw_points_from_labelme_with_data(image.permute(1, 2, 0).cpu().numpy(), label)

    image, label = original_ds[0]
    image = train_transform1(image)
    print(type(label))
    print(label.shape)
    print(image.shape)
    draw_points_from_labelme_with_data(image.permute(1, 2, 0).cpu().numpy(), label)


def imgaug_test():
    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    kps = KeypointsOnImage([
        Keypoint(x=65, y=100),
        Keypoint(x=75, y=200),
        Keypoint(x=100, y=100),
        Keypoint(x=200, y=80)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
        iaa.Affine(
            rotate=30,
            scale=(0.5, 0.7)
        ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    ])

    # Augment keypoints and images.
    image_aug, kps_aug = seq(image=image, keypoints=kps)
    print(type(image_aug))
    print(type(kps_aug))
    xy_array = kps_aug.to_xy_array()
    print(type(xy_array))
    draw_points_from_labelme_with_data(image_aug, xy_array)

    # print coordinates before/after augmentation (see below)
    # use after.x_int and after.y_int to get rounded integer coordinates
    for i in range(len(kps.keypoints)):
        before = kps.keypoints[i]
        after = kps_aug.keypoints[i]
        print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
            i, before.x, before.y, after.x, after.y)
        )

    # image with keypoints before/after augmentation (shown below)
    image_before = kps.draw_on_image(image, size=7)
    ia.imshow(image_before)
    image_after = kps_aug.draw_on_image(image_aug, size=7)
    ia.imshow(image_after)



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
        ], p=1)

    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )



keypoints_classes_ids2names = {0: 'Head', 1: 'Tail'}

def visualize(image, keypoints, image_original=None, keypoints_original=None):
    fontsize = 18
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40,40))
        plt.imshow(image)

    else:

        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)

from .dset import kps_key
import matplotlib.patches as patches

def visualize_augmentation(image, bboxes_augmented, keypoints_augmented, image_original, bboxes_original, keypoints_original):

    # print("keypoints_augmented", keypoints_augmented)
    # print("keypoints_original", keypoints_original)
    # print("bboxes_augmented", bboxes_augmented)
    # print("bboxes_original", bboxes_original)
    # Create a figure with two subplots

    # fig, axe = plt.subplots(1, 1, figsize=(40, 20))
    # axe.imshow(image_original)

    # # Define the points
    # points = keypoints_original
    # O, M, T, HP = points

    # # Calculate the equation of line L (going through O and M)
    # slope_L = (M[1] - O[1]) / (M[0] - O[0])
    # intercept_L = O[1] - slope_L * O[0]

    # # Calculate the x-coordinate of point N (intersection of L and vertical line through T)
    # x_N = T[0]
    # y_N = slope_L * x_N + intercept_L
    # N = np.array([x_N, y_N])

    # # Calculate the distances
    # dist_NT = np.linalg.norm(N - T)
    # dist_NHP = np.linalg.norm(N - HP)

    # # Calculate the ratio
    # ratio = dist_NT / dist_NHP

    # # Draw the lines and points on the axes
    # axe.plot([O[0], M[0]], [O[1], M[1]], label='Line L')
    # axe.plot([T[0], N[0]], [T[1], N[1]], label='Line TN')
    # axe.plot([N[0], HP[0]], [N[1], HP[1]], label='Line NH')
    # axe.scatter(points[:, 0], points[:, 1], label='Points', color='red')

    # axe.legend()
    # axe.invert_yaxis()  # Invert y-axis to match image coordinates

    # plt.show()


    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the original image with keypoints
    axes[0].imshow(image_original)
    # axes[0].scatter(keypoints_original[:, 0], keypoints_original[:, 1], c='r', marker='o')
    axes[0].scatter(keypoints_original[:, 0], keypoints_original[:, 1], s=4, c='red', marker='o')
    for i, label in enumerate(kps_key):
        axes[0].annotate(label, (keypoints_original[i][0], keypoints_original[i][1]), textcoords="offset points", xytext=(0,10), ha='center')

    for box in bboxes_original:
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)

    axes[0].set_title('Original Image with Keypoints')


    # Plot the augmented image with keypoints
    axes[1].imshow(image)
    # axes[1].scatter(keypoints_augmented[:, 0], keypoints_augmented[:, 1], c='r', marker='o')
    axes[1].scatter(keypoints_augmented[:, 0], keypoints_augmented[:, 1], s=4, c='red', marker='o')
    for i, label in enumerate(kps_key):
        axes[1].annotate(label, (keypoints_augmented[i][0], keypoints_augmented[i][1]), textcoords="offset points", xytext=(0,10), ha='center')

    for box in bboxes_augmented:
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
    axes[1].set_title('Augmented Image with Keypoints')


    # Show the plot
    plt.show()




import sys


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
        image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        

        image_original = (batch[2][0].permute(1, 2, 0).numpy() * 255).astype(
            np.uint8)
        visualize_augmentation(image, bboex_transformed, keypoint_transformed, image_original,
                               bboex_original, keypoint_original)


        # break
        # image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(
        #     np.int32).tolist()
        # print(type(image), image.shape)

        # keypoints = []
        # for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(
        #         np.int32).tolist():
        #     keypoints.append([kp[:2] for kp in kps])

        # image_original = (batch[2][0].permute(1, 2, 0).numpy() * 255).astype(
        #     np.uint8)
        # print(type(image_original), image_original.shape)
        # keypoints_original = batch[3][0]['keypoints'].detach().cpu().numpy(
        # ).astype(np.int32).tolist()
        # bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(
        #     np.int32).tolist()
        # print(bboxes)
        # keypoints_original = [[
        #     keypoints_original[i], keypoints_original[i + 1]
        # ] for i in range(0, len(keypoints_original), 2)]

        # # for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        # #     keypoints_original.append([kp for kp in kps])
        # print(type(keypoints), type(keypoints_original))

        # keypoints_m_draw = np.array(
        #     [point for sublist in keypoints for point in sublist])
        # keypoints_original_m_draw = np.array(
        #     [point for sublist in keypoints_original for point in sublist])
        # print("------------------------")
        # print("bboxes", bboxes)
        # print("bboxes_original", bboxes_original)
        # print("keypoints_m_draw", keypoints_m_draw)
        # print("keypoints_original_m_draw", keypoints_original_m_draw)

        # param like this

        # bboxes [[872, 807, 1162, 944], [872, 807, 1162, 944]]
        # bboxes_original [[841, 834, 1170, 928], [841, 834, 1170, 928]]
        # keypoints_m_draw [[ 891  906]
        # [ 921  869]
        # [1067  867]
        # [1152  872]]
        # keypoints_original_m_draw [[ 851  880]
        # [ 897  844]
        # [1065  885]
        # [1160  918]]
        # keypoints_augmented [[ 891  906]
        # [ 921  869]
        # [1067  867]
        # [1152  872]]
        # keypoints_original [[ 851  880]
        # [ 897  844]
        # [1065  885]
        # [1160  918]]
        # visualize_augmentation(image, bboxes, keypoints_m_draw, image_original,
        #    bboxes_original, keypoints_original_m_draw)








    # draw_points_from_labelme_with_data(image_original, keypoints_original_m_draw)
    # draw_points_from_labelme_with_data(image, keypoints_m_draw)



    # print(keypoints_original)
    # print(keypoints)
    # visualize(image, keypoints, image_original, keypoints_original)
    # for image, item_info_dict, image_original, item_info_dict_original in batch:

    #     print(image.shape)
    #     print(item_info_dict)
    #     image = (image.permute(1,2,0).numpy() * 255).astype(np.uint8)
    #     keypoints = []
    #     for kps in item_info_dict['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    #         keypoints.append([kp[:2] for kp in kps])

    #     image_original = (batch[2][0].permute(1,2,0).numpy() * 255).astype(np.uint8)

    #     keypoints_original = []
    #     for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    #         keypoints_original.append([kp[:2] for kp in kps])

    #     visualize(image, keypoints, image_original, keypoints_original)

    #     break


def augmentation_test():
    print(augmentation_test.__name__)

    dset = XRayDataset('data_300', transform=train_transform(), demo=True)

    dloader = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    start_time = time.time()

    visualize_test(dloader)


    # for batch_ndx, (images, labels) in enumerate(dloader):
    #     if batch_ndx > len(dloader) / 2:
    #         break
    #     print(type(images))
    #     print("(images, labels) in enumerate(dloader):", images.shape)
    #     item_ndx = labels["item_ndx"]
    #     keypoints = labels["keypoints"]

    #     print(item_ndx[0], keypoints[0])


    #     plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
    #     plt.axis('off')  # Turn off axis
    #     plt.show()
    #     break
    #     pass


def test_string_to_tensor():
    original_str = "Hello, world!"

    # Convert the string to a PyTorch tensor
    tensor = torch.tensor([ord(char) for char in original_str])

    # Convert the tensor back to a string
    converted_str = ''.join([chr(int(i)) for i in tensor])

    # Print the converted string
    print("Original String:", original_str)
    print("tensor:", tensor)
    print("Converted String:", converted_str)

from torchvision import transforms
from torchvision.datasets import ImageFolder
from .utils import delete_json_files
from .utils import move_images_to_dir


def dset_test():
    print(dset_test.__name__)


    dset = XRayDataset('data_300', transform=train_transform(), demo=True)

    dloader = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    iterator = iter(dloader)
    batch = next(iterator)
    # print("Original targets:\n", batch[3], "\n\n")
    # print("Transformed targets:\n", batch[1])




if __name__ == "__main__":
    print(__name__)
    # test_image_after_augmentation_with_transforms()
    # test1()

    # affine_test()
    # imgaug_test()

    # imgaug_test()

    # rename_label_files()

    # label_to_csv('data_300', 'data_300/label_dir')

    # delete_json_files('data_300')

    # move_images_to_dir('data_300', 'data_300/images')

    # cache_test()

    augmentation_test()
    # print(str(5) + "_epochs--" + datetime.datetime.now().strftime("%Y_%m_%d--%H:%M:%S") + "-weights.pth")

    # handle_raw_data_dir('data_300')

    # dset_test()


    # print(torch.cuda.device_count())
