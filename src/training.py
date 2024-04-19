import os
import sys


import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
import argparse
import datetime
import warnings

from .utils import collate_fn
from .utils import visualize_prediction_comparision
from .dset import XRayDataset
from .engine import train_one_epoch, evaluate
from util.logconf import logging
from .dset import KPS_KEY
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"

BATCH_SIZE = 2

class TrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()

        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=BATCH_SIZE,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=10,
            type=int,
        )
        parser.add_argument('--num-keypoints',
            help='Number of keypoints to detect',
            default=len(KPS_KEY),
            type=int,
        )


        self.cli_args = parser.parse_args(sys_argv)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0


        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    def init_model(self, num_keypoints, weights_path=None):
        anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                    pretrained_backbone=True,
                                                                    num_keypoints=num_keypoints,
                                                                    num_classes = 2, # Background is the first class, object is the second class
                                                                    rpn_anchor_generator=anchor_generator)
        if weights_path:
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model


    def train_transform(self):
        return A.Compose([
            A.Sequential([
                A.SafeRotate(limit=30, p=1), # Random rotation of an image by 90 degrees zero or more times
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.HorizontalFlip(p=0.5),  # Randomly flip the image horizontally
                A.VerticalFlip(p=0.5),  # Randomly flip the image vertically
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=50, shift_limit_x=0.2, shift_limit_y=0.2, p=0.5),  # Randomly shift, scale, and rotate the image
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ], p=1)
        ],
        keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
        )

    def init_train_dl(self):
        train_ds = XRayDataset(
            mode='train',
            transform=self.train_transform()
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn=collate_fn,
            shuffle=True
        )

        return train_dl

    def init_eval_dl(self):
        val_ds = XRayDataset(
            mode='eval'
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn=collate_fn,
            shuffle=True
        )

        return val_dl

    def init_test_dl(self):
        test_ds = XRayDataset(
            mode='test'
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn=collate_fn,
            shuffle=True
        )

        return test_dl

    def init_writer(self, writer_label):
        log_dir = os.path.join('runs', self.time_str)

        self.trn_writer = SummaryWriter(
            log_dir=log_dir + '-trn_cls-' + writer_label)
        self.val_writer = SummaryWriter(
            log_dir=log_dir + '-val_cls-' + writer_label)



    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        train_dl = self.init_train_dl()
        eval_dl = self.init_eval_dl()

        model = self.init_model(num_keypoints=self.cli_args.num_keypoints)

        
        params = [p for p in model.parameters() if p.requires_grad]
        writer_label = f'{self.cli_args.batch_size}bth_{self.cli_args.epochs}epochs_'

        # SGD
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)
        writer_label += 'SGD'
        
        # Adam
        # optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)
        # writer_label += 'Adam'

        self.init_writer(writer_label)
        trn_writer = self.trn_writer
        val_writer = self.val_writer
        
        num_epochs = self.cli_args.epochs

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(eval_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            train_one_epoch(model, optimizer, train_dl, self.device, epoch_ndx, print_freq=10, writer=trn_writer)
            lr_scheduler.step()
            evaluate(model, eval_dl, self.device, writer=val_writer, epoch=epoch_ndx)


        trn_writer.close()
        # Save model weights after training
        torch.save(model.state_dict(), str(num_epochs) + "_epochs-" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "-weights.pth")

        print('s')


    def test(self):
        test_dl = self.init_test_dl()
        model = self.init_model(num_keypoints=self.cli_args.num_keypoints)
        model.load_state_dict(torch.load('10_epochs--20240419_115217-weights.pth'))

        count = 0
        for images, targets in test_dl:
            images = list(image.to(self.device) for image in images)
            box_groundtruth = targets[0]["boxes"].numpy().astype(int)
            kps_groundtruth = targets[0]["keypoints"][..., :2].numpy()[0].astype(int)
            
            with torch.no_grad():
                model.to(self.device)
                model.eval()
                output = model(images)

            image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            scores = output[0]['scores'].detach().cpu().numpy()

            high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
            post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

            # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
            # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
            # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])

            bboxes = []
            for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                bboxes.append(list(map(int, bbox.tolist())))
            if len(keypoints) == 0:
                continue
            keypoints = keypoints[0]

            kps_predicted = np.array(keypoints)
            # Swap elements where the denominator is smaller
            swap_mask = kps_groundtruth < kps_predicted
            kps_divided = np.where(swap_mask, kps_groundtruth, kps_predicted) / np.where(swap_mask, kps_predicted, kps_groundtruth)
            # Compute the average of the sum of coordinates
            flat_data = kps_divided.flatten()
            # Compute the average
            average_value = np.mean(flat_data)
            print("Average similarity:", average_value)
            if average_value > 0.95:
                count += 1
                visualize_prediction_comparision(image, np.array(bboxes), np.array(keypoints), box_groundtruth, kps_groundtruth)
            
        print("avg_simi > 0.99:", count, "all:", len(test_dl))

if __name__ == '__main__':
    TrainingApp().main()
    # TrainingApp().test()
