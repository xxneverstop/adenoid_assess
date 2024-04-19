import json
import re
import random
import glob
import shutil
import csv

import matplotlib.pyplot as plt

from src.dset import KPS_KEY

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
    for i, label in enumerate(KPS_KEY):
        axes.annotate(label, (keypoints_original[i][0], keypoints_original[i][1]), textcoords="offset points", xytext=(0,10), ha='center')
    axes.scatter(keypoints_predicted[:, 0], keypoints_predicted[:, 1], s=8, c='green', marker='o')
    axes.set_title('visualize_prediction_comparision')
    plt.show()



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


def handle_raw_data_dir(raw_data_dir):
    print("handle_raw_data_dir")
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

        if not new_name:
            print(subdir)
            sys.exit()
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





import sys
def label_to_csv(label_dir, csv_output_dir):
    print(label_to_csv.__name__)

    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir, exist_ok=True)

    csv_file_name = 'kps_data.csv'
    csv_dump_path = os.path.join(csv_output_dir, csv_file_name)

    with open(csv_dump_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header = ['series_id']
        for i in range(len(KPS_KEY)):
            header.extend([f'{KPS_KEY[i]}_x', f'{KPS_KEY[i]}_y'])
        writer.writerow(header)

        json_files = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir) if filename.endswith('.json')]

        count = 0   

        # ensure the key matches to the header
        for label_file_path in json_files:
            with open(label_file_path, 'r') as f:
                label_data = json.load(f)

            _, filename = os.path.split(label_file_path)
            series_uid = filename[:-5]
            row = [series_uid]

            # change along with the num_keypoints
            for i in range(2 * len(KPS_KEY)):
                row.append([])
            pass_flag = False
            for idx, shape in enumerate(label_data['shapes']):
                label = shape['label']
                
                try:
                    ndx = KPS_KEY.index(label)
                except:
                    print("An exception occurred, ValueError", series_uid)
                    
                if idx != ndx:  # 1202214461-0001_1
                    pass_flag = True
                    # index not match
                    # print('index of label does not match')
                    break

                assert shape['shape_type'] == 'point'

                points = shape['points']
                assert len(points) == 1


                row[ndx * 2 + 1], row[ndx * 2 + 2] = points[0][0], points[0][1]

            for i in range(len(row)):
                if not row[i]:
                    pass_flag = True
                    # print('num of label does not match')
                    break

            if pass_flag:
                print('idx no match', series_uid, 'pass this sample, continue')
                continue
            writer.writerow(row)
            count += 1
            
        print(f"kps data has been dumped. {count} rows dumped, all: {len(json_files)}")




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







