import math
import sys
import time
import numpy as np


import torch
import torchvision.models.detection.mask_rcnn
import src.utils as utils
import torch.nn.functional as F
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,
                    writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    count = 1
    epoch_avg_train_loss_sum = 0

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
        # print("loss dict while training, loss_dict = model(images, targets), batch_size = 3:\n", loss_dict)

        # loss_dict['loss_classifier']: Classification loss. This measures how well the model is classifying the objects present in the image.
        # loss_dict['loss_box_reg']: Bounding box regression loss. This measures how accurate the model's predicted bounding boxes are compared to the ground truth bounding boxes.
        # loss_dict['loss_keypoint']: Keypoint detection loss. This measures the accuracy of the model's predicted keypoints compared to the ground truth keypoints.
        # loss_dict['loss_objectness']: Objectness loss. This measures how well the model is distinguishing between object and background regions.
        # loss_dict['loss_rpn_box_reg']: Region Proposal Network (RPN) bounding box regression loss. This measures the accuracy of the RPN's predicted bounding boxes compared to the ground truth bounding boxes.

        # print("loss_dict", loss_dict)

        x_idx = (epoch - 1) * len(data_loader) + count
        losses = sum(loss for loss in loss_dict.values())
        writer.add_scalar('train_loss_sum', losses, x_idx)

        writer.add_scalar('train_loss_keypoint',
                          loss_dict['loss_keypoint'].item(), x_idx)
        writer.add_scalar('train_loss_rpn_box_reg',
                          loss_dict['loss_rpn_box_reg'].item(), x_idx)

        count += 1
        epoch_avg_train_loss_sum += losses

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


    if count > 1:
        epoch_avg_train_loss_sum = epoch_avg_train_loss_sum / (count - 1)
        writer.add_scalar('epoch_avg_loss', epoch_avg_train_loss_sum, epoch)


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


def get_loss_from_output(targets, output):
    box_groundtruth = targets["boxes"].numpy().astype(int)
    kps_groundtruth = targets["keypoints"][..., :2].numpy()[0].astype(int)
    # print("Predictions: \n", output)
    scores = output['scores'].detach().cpu().numpy()

    high_scores_idxs = np.where(scores > 0)[0].tolist() # Indexes of boxes with scores > 0.7
    post_nms_idxs = torchvision.ops.nms(output['boxes'][high_scores_idxs], output['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

    # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
    # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
    # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

    keypoints = []
    for kps in output['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])

    bboxes = []
    for bbox in output['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))

    if len(keypoints) == 0:
        return 0, 0
    keypoints = keypoints[0]
    # print("predicted bboxes:\n", np.array(bboxes))
    # print("predicted keypoints:\n", np.array(keypoints))


    # similarity
    kps_predicted = np.array(keypoints)
    # Swap elements where the denominator is smaller
    swap_mask = kps_groundtruth < kps_predicted
    kps_divided = np.where(swap_mask, kps_groundtruth, kps_predicted) / np.where(swap_mask, kps_predicted, kps_groundtruth)
    # print("kps_divided", kps_divided)
    # Compute the average of the sum of coordinates
    flat_data = kps_divided.flatten()
    # Compute the average
    average_similarity = np.mean(flat_data)
    # print("Average similarity:", average_value)


    # distances
    # Ensure both arrays have the same shape
    assert kps_groundtruth.shape == kps_predicted.shape, "Arrays must have the same shape"
    # Initialize an array to store distances
    distances = np.zeros(kps_groundtruth.shape[0])
    # Compute distances for each pair of points
    for i in range(kps_groundtruth.shape[0]):
        distances[i] = np.linalg.norm(kps_groundtruth[i] - kps_predicted[i])

    average_distance = np.mean(distances)

    return average_similarity, average_distance


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

    epoch_avg_val_similarity = 0
    epoch_avg_val_distance = 0

    for images, targets in metric_logger.log_every(data_loader, 50, header):
        images = list(img.to(device) for img in images)

        # print("len(images), while validation:", len(images))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        # print("output of validation, batch_size = 1:\n", outputs)

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

            similarity, distance = get_loss_from_output(target, output)

            epoch_avg_val_similarity += similarity
            epoch_avg_val_distance += distance

            if similarity == 0 and distance == 0:
                continue
            writer.add_scalar('val_kps_avg_similarity', similarity, x_idx)
            writer.add_scalar('val_kps_avg_distance', distance, x_idx)

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

    
    if count > 1:
        epoch_avg_val_similarity = epoch_avg_val_similarity / (count - 1)
        writer.add_scalar('epoch_avg_val_similarity', epoch_avg_val_similarity, epoch)
        epoch_avg_val_distance = epoch_avg_val_distance / (count - 1)
        writer.add_scalar('epoch_avg_val_distance', epoch_avg_val_distance, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
