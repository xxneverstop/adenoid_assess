# import math
# import sys
# import time

# import torch
# import torchvision.models.detection.mask_rcnn
# import src.utils as utils
# from .coco_eval import CocoEvaluator
# from .coco_utils import get_coco_api_from_dataset
# from util.logconf import logging
# from util.util import enumerateWithEstimate

# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):


#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     header = f"Epoch: [{epoch}]"

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=warmup_factor, total_iters=warmup_iters
#         )

#     batch_iter = enumerateWithEstimate(
#             data_loader,
#             "E{} Training".format(epoch),
#             start_ndx=data_loader.num_workers,
#         )

#     for batch_ndx, (images, targets) in batch_iter:
#         images = list(image.to(device) for image in images)
#         # print("targets = [{k: v.to(device) for k, v in t.items()} for t in targets]\n", targets)
#         # sys.exit()
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         loss_dict = model(images, targets)

#         losses = sum(loss for loss in loss_dict.values())

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

#         loss_value = losses_reduced.item()

#         if not math.isfinite(loss_value):
#             print(f"Loss is {loss_value}, stopping training")
#             print(loss_dict_reduced)
#             sys.exit(1)

#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()

#         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     return metric_logger


# def _get_iou_types(model):
#     model_without_ddp = model
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model_without_ddp = model.module
#     iou_types = ["bbox"]
#     if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
#         iou_types.append("segm")
#     if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
#         iou_types.append("keypoints")
#     return iou_types


# @torch.inference_mode()
# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Test:"

#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)

#     for images, targets in metric_logger.log_every(data_loader, 100, header):
#         images = list(img.to(device) for img in images)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(images)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator


import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import src.utils as utils
import torch.nn.functional as F
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    count = 1


    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    for images, targets in metric_logger.log_every(data_loader, print_freq,
                                                   header):
        images = list(image.to(device) for image in images)
        # print("targets:", targets)
        # sys.exit()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # loss_dict['loss_classifier']: Classification loss. This measures how well the model is classifying the objects present in the image.
        # loss_dict['loss_box_reg']: Bounding box regression loss. This measures how accurate the model's predicted bounding boxes are compared to the ground truth bounding boxes.
        # loss_dict['loss_keypoint']: Keypoint detection loss. This measures the accuracy of the model's predicted keypoints compared to the ground truth keypoints.
        # loss_dict['loss_objectness']: Objectness loss. This measures how well the model is distinguishing between object and background regions.
        # loss_dict['loss_rpn_box_reg']: Region Proposal Network (RPN) bounding box regression loss. This measures the accuracy of the RPN's predicted bounding boxes compared to the ground truth bounding boxes.

        # print("loss_dict", loss_dict)

        x_idx = (epoch - 1) * len(data_loader) + count
        losses = sum(loss for loss in loss_dict.values())
        writer.add_scalar('train_loss_sum', losses, x_idx)
        count += 1


        # print("losses:", losses)
        # reduce losses over all GPUs for logging purposes
        # if one GPU, loss_dic does not change
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, writer, epoch):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    count = 1
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{
            k: v.to(cpu_device)
            for k, v in t.items()
        } for t in outputs]


        for target, output in zip(targets, outputs):
            
            keypoints_gt = target["keypoints"]
            predicted_keypoints = output["keypoints"]
            loss = F.mse_loss(predicted_keypoints, keypoints_gt)
            validation_loss = loss.item()
            x_idx = (epoch - 1) * len(data_loader) + count
            writer.add_scalar('val_loss', validation_loss, x_idx)
            count += 1




        
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
