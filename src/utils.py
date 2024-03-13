import os
import json
import numpy as np
import cv2
import re
import random
import glob
import shutil
import time
import csv
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.dset import kps_key


def visualize_prediction_comparision(image, bboxes_predicted, keypoints_predicted, bboxes_original, keypoints_original):

    # print("keypoints_predicted", keypoints_predicted)
    # print("keypoints_original", keypoints_original)
    # print("bboxes_predicted", bboxes_predicted)
    # print("bboxes_original", bboxes_original)
    # Create a figure with two subplots
    # fig, axes = plt.subplots(1, 2, figsize=(20, 10))      # two images one row
    fig, axes = plt.subplots(1, 1, figsize=(40, 20))
    axes.imshow(image)
    axes.scatter(keypoints_original[:, 0], keypoints_original[:, 1], s=8, c='red', marker='o')
    for i, label in enumerate(kps_key):
        axes.annotate(label, (keypoints_original[i][0], keypoints_original[i][1]), textcoords="offset points", xytext=(0,10), ha='center')
    axes.scatter(keypoints_predicted[:, 0], keypoints_predicted[:, 1], s=8, c='green', marker='o')
    axes.set_title('visualize_prediction_comparision')
    plt.show()


    # for i in range(4):
    #     fig, axes = plt.subplots(1, 1, figsize=(40, 20))
    #     axes.imshow(image)
    #     axes.scatter(keypoints_original[i][0], keypoints_original[i][1], s=8, c='red', marker='o')
    #     axes.scatter(keypoints_predicted[i][0], keypoints_predicted[i][1], s=8, c='green', marker='o')
    #     axes.set_title('visualize_prediction_comparision')
    #     plt.show()


    # fig, axe = plt.subplots(1, 1, figsize=(40, 20))
    # axe.imshow(image)

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



    




    # # Plot the original image with keypoints
    # axes.imshow(image)
    # # axes.scatter(keypoints_original[:, 0], keypoints_original[:, 1], c='r', marker='o')
    # # axes.scatter(keypoints_original[:, 0], keypoints_original[:, 1], s=8, c='red', marker='o')
    # axes.scatter(keypoints_original[0][0], keypoints_original[0][1], s=8, c='red', marker='o')

    # # add label of kps
    # # for i, label in enumerate(kps_key):
    # #     axes.annotate(label, (keypoints_original[i][0], keypoints_original[i][1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    # # add box
    # # for box in bboxes_original:
    # #     x, y, w, h = box
    # #     rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
    # #     axes.add_patch(rect)


    # # axes.scatter(keypoints_predicted[:, 0], keypoints_predicted[:, 1], s=8, c='green', marker='o')
    # axes.scatter(keypoints_predicted[0][0], keypoints_predicted[0][1], s=8, c='red', marker='o')

    # # for i, label in enumerate(kps_key):
    # #     axes.annotate(label, (keypoints_predicted[i][0], keypoints_predicted[i][1]), textcoords="offset points", xytext=(0,10), ha='center')


    # axes.set_title('visualize_prediction_comparision')





    # # Plot the original image with keypoints
    # axes[0].imshow(image)
    # # axes[0].scatter(keypoints_original[:, 0], keypoints_original[:, 1], c='r', marker='o')
    # axes[0].scatter(keypoints_original[:, 0], keypoints_original[:, 1], s=4, c='red', marker='o')

    # # add label of kps
    # # for i, label in enumerate(kps_key):
    # #     axes[0].annotate(label, (keypoints_original[i][0], keypoints_original[i][1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    # # add box
    # # for box in bboxes_original:
    # #     x, y, w, h = box
    # #     rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
    # #     axes[0].add_patch(rect)


    # axes[0].scatter(keypoints_predicted[:, 0], keypoints_predicted[:, 1], s=4, c='green', marker='o')
    # # for i, label in enumerate(kps_key):
    # #     axes[0].annotate(label, (keypoints_predicted[i][0], keypoints_predicted[i][1]), textcoords="offset points", xytext=(0,10), ha='center')


    # axes[0].set_title('visualize_prediction_comparision')








    # # Plot the augmented image with keypoints
    # axes[1].imshow(image)
    # # axes[1].scatter(keypoints_predicted[:, 0], keypoints_predicted[:, 1], c='r', marker='o')
    # axes[1].scatter(keypoints_predicted[:, 0], keypoints_predicted[:, 1], s=4, c='red', marker='o')
    # for i, label in enumerate(kps_key):
    #     axes[1].annotate(label, (keypoints_predicted[i][0], keypoints_predicted[i][1]), textcoords="offset points", xytext=(0,10), ha='center')

    # for box in bboxes_predicted:
    #     x, y, w, h = box
    #     rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
    #     axes[1].add_patch(rect)
    # axes[1].set_title('Augmented Image with Keypoints')


    # Show the plot
    # plt.show()


def predict_coordinates(model, image_path, transform):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        if transform:
            image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        outputs = model(image)
        predicted_coordinates = outputs.squeeze().numpy()
    return predicted_coordinates


def train_test():
    class XRayDataset(Dataset):
        def __init__(self, data_dir, transform=None, max_num_points=20):
            self.data_dir = data_dir
            self.transform = transform
            self.max_num_points = max_num_points
            self.image_paths = sorted([os.path.join(data_dir, 'figure', filename) for filename in
                                       os.listdir(os.path.join(data_dir, 'figure'))])
            self.label_paths = sorted(
                [os.path.join(data_dir, 'data', filename) for filename in os.listdir(os.path.join(data_dir, 'data'))])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            label_path = self.label_paths[idx]

            # Load image
            image = Image.open(image_path).convert('RGB')

            # Load label data from JSON file
            with open(label_path, 'r') as f:
                label_data = json.load(f)

            # Extract coordinates from the first shape (assuming only one shape per image)
            points = label_data['shapes'][0]['points']
            num_points = len(points)

            # Pad or truncate points to fixed size
            if num_points < self.max_num_points:
                pad_points = points + [[0, 0]] * (self.max_num_points - num_points)
                label = np.array(pad_points, dtype=np.float32)
            else:
                label = np.array(points[:self.max_num_points], dtype=np.float32)

            if self.transform:
                image = self.transform(image)

            return image, label

    # Define transformations for data augmentation and normalization
    transform = transforms.Compose([transforms.Resize((128, 128)),  # Resize images to 128x128
                                    transforms.ToTensor(),  # Convert PIL image to tensor
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    # Normalize pixel values
                                    ])

    # Create dataset and dataloaders
    data_dir = "dataset"
    dataset = XRayDataset(data_dir, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Define the CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 16 * 16, 64)
            self.fc2 = nn.Linear(64, 2)  # Output layer with 2 nodes for x and y coordinates

        def forward(self, x):
            print(x.size())
            x = self.pool(torch.relu(self.conv1(x)))
            print(x.size())
            x = self.pool(torch.relu(self.conv2(x)))
            print(x.size())
            x = self.pool(torch.relu(self.conv3(x)))
            print(x.size())
            x = torch.flatten(x, 1)
            print(x.size())
            x = torch.relu(self.fc1(x))
            print(x.size())
            x = self.fc2(x)
            print(x.size())
            return x

    # Initialize the model, loss function, and optimizer
    print('about to create model')
    model = CNN()
    print('create model')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Training
        model.train()
        print('after train')
        running_loss = 0.0
        for images, labels in train_loader:
            print('images:', images.size())
            print('after for images, labels in train_loader:')
            optimizer.zero_grad()
            outputs = model(images)
            print('output:', outputs.size())
            print('labels:', labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_dataset)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

    # After training, you can use the model for inference
    # For example, to predict coordinates on a new image:
    # new_image = load_and_preprocess_new_image('path_to_image.jpg')
    # new_image = torch.tensor(new_image).unsqueeze(0)  # Add batch dimension
    # predicted_coordinates = model(new_image)
    new_image_path = "D:\code\python\Adenoid Hypertrophy Assessment\30_items\dataset\test\Eddie wang.JPG"
    predicted_coordinates = predict_coordinates(model, new_image_path, transform)
    print("Predicted Coordinates:", predicted_coordinates)



# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# (1687, 2050, 3), (21, 2) 
def draw_points_from_labelme_with_data(image, label):
    # Create a new plot with a larger window
    plt.figure(figsize=(10, 8))  # Adjust width and height as needed

    # Display the image
    plt.imshow(image)

    for point in label:
        # Plot each point
        x_values, y_values = point
        plt.scatter(x_values, y_values, color='red', marker='o')  # Markers for points
    plt.show()
    



def draw_points_from_labelme(image_file_path, json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Read the image
    image = plt.imread(image_file_path)
    print(type(image))
    print(image.shape)
    # Create a new plot with a larger window
    plt.figure(figsize=(10, 8))  # Adjust width and height as needed

    # Display the image
    plt.imshow(image)

    # Iterate over each polygon in the data
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            # Extract points
            points = shape['points']

            # Plot each point
            x_values, y_values = zip(*points)
            plt.scatter(x_values, y_values, color='red', marker='o')  # Markers for points

    # Show the plot
    plt.show()


def rotate_point(point, angle, center):
    # angle = -angle
    x, y = point
    center_x, center_y = center
    translated_point = (x - center_x, center_y - y)
    rotated_x = translated_point[0] * np.cos(np.radians(angle)) - translated_point[1] * np.sin(np.radians(angle))
    rotated_y = translated_point[0] * np.sin(np.radians(angle)) + translated_point[1] * np.cos(np.radians(angle))
    rotated_point = (rotated_x, rotated_y)
    translated_rotated_point = (rotated_point[0] + center_x, center_y - rotated_point[1])
    return translated_rotated_point


def rotate_labelme_annotations(json_file_path, dump_path, angle):
    # Load the LabelMe JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Get the image size
    img_height, img_width = data['imageHeight'], data['imageWidth']

    # Calculate the center of rotation
    center = (img_width / 2, img_height / 2)

    # Iterate over each shape in the JSON file
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            # Rotate each point in the shape
            rotated_points = [rotate_point(point, angle, center) for point in shape['points']]

            # Update the rotated points in the shape
            shape['points'] = rotated_points

    # Update image rotation
    # data['imageRotation'] = angle

    # Save the rotated data back to a JSON file
    output_json_file_path = dump_path
    with open(output_json_file_path, 'w') as file:
        json.dump(data, file, indent=2)  # return data


def has_three_channels(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            print("Error: Unable to open image file")
            return False
        print(img.shape)
        print(torch.cuda.device_count())
        return img.shape[2] == 3
    except Exception as e:
        print(f"Error: {e}")
        return False


def rotate_with_PIL(in_path, out_path):
    # Load the image
    image_path = in_path
    image = Image.open(image_path)

    # Rotate the image by 15 degrees
    rotated_image = image.rotate(15, expand=True)

    # Save the rotated image
    rotated_image.save(out_path)


def rotate_with_cv2(in_path, out_path, angle):
    # Load the image
    image_path = in_path
    image = cv2.imread(image_path)

    # Rotate the image by 15 degrees
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Save the rotated image
    cv2.imwrite(out_path, rotated_image)


def sync_label_rotation(image_path, label_path, angle):
    jpg_directory, jpg_filename = os.path.split(image_path)
    jpg_name, jpg_extension = os.path.splitext(jpg_filename)

    json_directory, json_filename = os.path.split(label_path)
    json_name, json_extension = os.path.splitext(json_filename)

    assert jpg_name == json_name

    rotated_jpg_name = f"{jpg_name}_rotate{angle}{jpg_extension}"
    rotated_json_name = f"{json_name}_rotate{angle}{json_extension}"

    jpg_rotated_path = os.path.join(jpg_directory, rotated_jpg_name)
    json_rotated_path = os.path.join(json_directory, rotated_json_name)

    # draw_points_from_labelme(image_path, label_path)

    rotate_with_cv2(image_path, jpg_rotated_path, angle=angle)
    rotate_labelme_annotations(label_path, json_rotated_path, angle=angle)

    # draw_points_from_labelme(jpg_rotated_path, json_rotated_path)


def test_sync_label_rotation():
    image_directory = 'dataset/image'
    label_directory = 'dataset/label'
    # Get list of JPG files sorted
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.jpg')])

    # Get list of JSON files sorted
    label_files = sorted([f for f in os.listdir(label_directory) if f.endswith('.json')])

    degrees_range_list = list(range(15, 40 + 1, 5))
    degrees_step = 5

    # Iterate over both lists simultaneously
    for image_file, label_file in zip(image_files, label_files):
        if len(image_file) != 9 or len(label_file) != 10:
            continue

        # Construct the full paths
        image_path = os.path.join(image_directory, image_file)
        label_path = os.path.join(label_directory, label_file)

        degree_range = random.choice(degrees_range_list)
        random_angle = get_random_degree(degree_range, degrees_step)
        if random_angle == 0:
            continue
        sync_label_rotation(image_path, label_path, angle=random_angle)

        # Print or process the paths as needed
        print("rotate image file:", image_path)
        print("rotate label file:", label_path)


def handle_decode_issue():
    # Directory containing JSON files
    directory = 'dataset/label'

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            # Read JSON file as text
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                json_text = file.read()

            # Modify "label" field to "m_label" in the JSON text
            # json_text = json_text.replace('"m_label"', '"label"')
            # Replace the value of "label" field with "m_label"
            modified_json_text = re.sub(r'"imagePath"\s*:\s*"[^"]*"', '"imagePath": "m_path"', json_text)
            # Load modified JSON data
            # Write modified JSON back to the file
            with open(filepath, 'w') as file:
                file.write(modified_json_text)

            print(f"Modified {filename} successfully.")

    print("All files modified.")


def process_filename():
    # Directory paths
    image_dir = 'dataset/image'
    label_dir = 'dataset/label'

    # Get list of image files sorted
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.JPG') or f.endswith('.jpg')])

    # Get list of label files sorted
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])

    # Ensure the number of image files matches the number of label files
    assert len(image_files) == len(label_files), "Number of image files does not match number of label files."

    # Rename image files and corresponding label files
    for i, (image_file, label_file) in enumerate(zip(image_files, label_files), start=1):
        # Ensure the names of JPG and JSON files match
        image_name, image_ext = os.path.splitext(image_file)
        label_name, label_ext = os.path.splitext(label_file)
        assert image_name == label_name, f"Name mismatch between {image_file} and {label_file}."

        # New filenames
        new_image_name = f'{i:05d}.jpg'
        new_label_name = f'{i:05d}.json'

        # Current file paths
        current_image_path = os.path.join(image_dir, image_file)
        current_label_path = os.path.join(label_dir, label_file)

        # New file paths
        new_image_path = os.path.join(image_dir, new_image_name)
        new_label_path = os.path.join(label_dir, new_label_name)

        # Rename image file
        os.rename(current_image_path, new_image_path)

        # Rename label file
        os.rename(current_label_path, new_label_path)

        print(f"Renamed {current_image_path} to {new_image_path}")
        print(f"Renamed {current_label_path} to {new_label_path}")

    print("All files renamed.")


# Function to delete files with specified pattern
def delete_files_with_len(directory, len_of_filename):
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if len(filename) != len_of_filename:
            file_path = os.path.join(directory, filename)
            # Delete the file
            os.remove(file_path)
            print(f"Deleted {file_path}")


def delete_test_file(image_dir, label_dir):
    # Directory paths
    image_directory = image_dir
    label_directory = label_dir

    # Delete files with the specified pattern in the image directory
    print("Deleting files in the image directory:")
    delete_files_with_len(image_directory, 9)

    # Delete files with the specified pattern in the label directory
    print("\nDeleting files in the label directory:")
    delete_files_with_len(label_directory, 10)

    print("\nDeletion completed.")


def get_random_degree(degree_range, degree_step):
    # Generate the list of degrees
    degrees = list(range(-degree_range, degree_range + 1, degree_step))
    # Randomly select and print a degree 10 times
    # for _ in range(10):
    #     selected_degree = random.choice(degrees)
    #     selected_idx = degrees.index(selected_degree)
    #     print("Selected degree:", selected_degree)
    #     print("Index of selected degree:", selected_idx)
    #     print()
    selected_degree = random.choice(degrees)
    return selected_degree


def handle_raw_data_dir(raw_data_dir):
    dir_path = raw_data_dir
    subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    
    for subdir in subdirs:
        # Split the subdir name at the first underscore
        parts = subdir.split('_', 1)
        new_name = ''
        if len(parts) > 1:
            # Extract the part after the first underscore
            new_name = parts[1]
            # Rename the subdir
            os.rename(os.path.join(dir_path, subdir), os.path.join(dir_path, new_name))
        assert new_name
        file_list = glob.glob(os.path.join(dir_path, new_name) + '/*/*.JPG')
        json_list = glob.glob(os.path.join(dir_path, new_name) + '/*/*.json')
        # rename them, jpg files may have more than one
        for jpg_file in file_list:
            jpg_dir, jpg_name = os.path.split(jpg_file)
            new_jpg_name = new_name + '_' + jpg_name[:-4] + '.jpg'
            os.rename(jpg_file, os.path.join(jpg_dir, new_jpg_name))
            print(new_jpg_name)
        for json_file in json_list:
            json_dir, json_name = os.path.split(json_file)
            new_json_name = new_name + '_' + json_name[:-5] + '.json'
            os.rename(json_file, os.path.join(json_dir, new_json_name))
            print(new_json_name)

    # move jpg to the dir_path
    

    label_dir = os.path.join(raw_data_dir, 'label_dir')
    os.makedirs(label_dir, exist_ok=True)

    # image_dir = os.path.join(raw_data_dir, 'image_dir')
    # os.makedirs(image_dir, exist_ok=True)

    jpg_list = glob.glob(dir_path + '/*/*/*.jpg')
    for x in jpg_list:
        shutil.move(x, dir_path)    

    json_list = glob.glob(dir_path + '/*/*/*.json')
    for x in json_list:
        shutil.move(x, dir_path)
        
    # delete the empty dir
    for dirpath, dirnames, filenames in os.walk(dir_path, topdown=False):
        # Check if the directory has any file
        if not filenames:
            # Remove the empty directory
            os.rmdir(dirpath)
            print(f"Removed empty directory: {dirpath}")



def rename_label_files():
    dir_label = 'dataset_600/label'
    dir_raw = 'raw_data_dir'
    image_files = sorted([f for f in os.listdir(dir_raw) if f.endswith('.JPG') or f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(dir_label) if f.endswith('.json')])
    assert len(image_files) == len(label_files), "Number of image files does not match number of label files."
    for i, (image_file, label_file) in enumerate(zip(image_files, label_files), start=1):
        image_name, image_ext = os.path.splitext(image_file)
        label_name, label_ext = os.path.splitext(label_file)

        current_label_path = os.path.join(dir_label, label_file)
        new_label_name = f'{image_name}.json'
        new_label_path = os.path.join(dir_label, new_label_name)
        os.rename(current_label_path, new_label_path)

        print(new_label_name, new_label_path)





from pathlib import Path
import sys

# dump keypoints in label json file into csv file
def label_to_csv(label_dir, csv_output_dir):
    print(label_to_csv.__name__)

    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir, exist_ok=True)

    csv_file_name = 'kps_data.csv'
    csv_dump_path = os.path.join(csv_output_dir, csv_file_name)

    with open(csv_dump_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header = ['series_id']
        for i in range(4):
            header.extend([f'{kps_key[i]}_x', f'{kps_key[i]}_y'])
        writer.writerow(header)

        json_files = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir) if filename.endswith('.json')]

        # ensure the key matches to the header
        for label_file_path in json_files:
            with open(label_file_path, 'r') as f:
                label_data = json.load(f)

            _, filename = os.path.split(label_file_path)
            series_uid = filename[:-5]
            row = [series_uid]
            for i in range(8):
                row.append([])
            pass_flag = False
            for idx, shape in enumerate(label_data['shapes']):
                label = shape['label']
                ndx = kps_key.index(label)

                if idx != ndx:  # 1202214461-0001_1
                    pass_flag = True
                    break

                assert shape['shape_type'] == 'point'

                points = shape['points']
                assert len(points) == 1

                row[ndx * 2 + 1], row[ndx * 2 + 2] = points[0][0], points[0][1]

            if pass_flag:
                print('idx no match', series_uid, 'pass this sample, continue')
                continue
            writer.writerow(row)
        print(f"kps data has been dumped.")



def delete_json_files(directory):
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Check if the file ends with '.json'
        if filename.endswith('.json'):
            # Full path of the file
            filepath = os.path.join(directory, filename)
            # Delete the file
            os.remove(filepath)
            print(f"Deleted file: {filepath}")



def move_images_to_dir(from_dir, to_dir):
    # Iterate over files in the source directory
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    for filename in os.listdir(from_dir):
        # Check if the file ends with '.jpg'
        if filename.endswith('.jpg'):
            # Full path of the source file
            source_filepath = os.path.join(from_dir, filename)
            # Full path of the destination file
            destination_filepath = os.path.join(to_dir, filename)
            # Move the file to the destination directory
            shutil.move(source_filepath, destination_filepath)
            print(f"Moved file: {source_filepath} to {destination_filepath}")


import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()   # not distributed system
    if world_size < 2:
        return input_dict
    with torch.inference_mode():    # if distributed, get averaged results
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time_cost: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                    "now: {now}"
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                            now=datetime.datetime.now()
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
