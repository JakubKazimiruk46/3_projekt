import os
import torch
import logging
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from torchvision import models
from config import Config

config = Config()
IOU_THRESHOLDS = [0.3]
class_names = config.class_names


def initialize_face_model(num_classes):
    """Initialize face detection model"""
    # SSDLite with MobileNetV3 - good for real-time face detection
    model = models.detection.ssdlite320_mobilenet_v3_large(
        pretrained=False,
        num_classes=num_classes + 1  # +1 for background class
    )

    return model


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for i, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
        valid_indices = [j for j, t in enumerate(targets) if len(t['boxes']) > 0]
        if not valid_indices:
            continue

        valid_images = [images[j] for j in valid_indices]
        valid_targets = [targets[j] for j in valid_indices]

        images_device = [img.to(device) for img in valid_images]
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in valid_targets]

        optimizer.zero_grad()
        try:
            loss_dict = model(images_device, targets_device)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
            num_batches += 1

            if i % 20 == 19 and num_batches > 0:
                avg_loss = running_loss / num_batches
                logging.info(f"  Batch {i + 1}, Avg Loss: {avg_loss:.4f}")

        except Exception as e:
            logging.error(f"Error in training batch {i}: {e}")
            continue

    return running_loss / max(num_batches, 1)


def evaluate_model(model, dataloader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=IOU_THRESHOLDS)
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            valid_indices = [j for j, t in enumerate(targets) if len(t['boxes']) > 0]
            if not valid_indices:
                continue

            valid_images = [images[j] for j in valid_indices]
            valid_targets = [targets[j] for j in valid_indices]

            images_device = [img.to(device) for img in valid_images]
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in valid_targets]

            try:
                predictions = model(images_device)

                model.train()
                loss_dict = model(images_device, targets_device)
                val_loss += sum(loss for loss in loss_dict.values()).item()
                val_batches += 1
                model.eval()

                predictions_cpu = [{k: v.cpu() for k, v in p.items()} for p in predictions]
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in valid_targets]
                metric.update(predictions_cpu, targets_cpu)

            except Exception as e:
                logging.error(f"Error in validation batch: {e}")
                continue

    try:
        map_value = metric.compute()['map'].item()
    except Exception as e:
        logging.warning(f"Failed to compute mAP: {e}")
        map_value = 0.0

    return val_loss / max(val_batches, 1), map_value


def save_model_checkpoint(model, optimizer, epoch, train_loss, val_loss, map_value, best_map, is_best):
    if is_best:
        checkpoint_path = "face_checkpoints/best_yolo_face_model.pth"
        logging.info(f"  New best model! mAP: {map_value:.4f}")
    else:
        checkpoint_path = f"face_checkpoints/yolo_face_model_epoch_{epoch + 1}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'map': map_value,
        'best_map': best_map,
        'class_names': class_names
    }, checkpoint_path)
    logging.info(f"  Model saved to {checkpoint_path}")


def train_face_model(model, train_dataloader, val_dataloader, num_epochs=30, device='cpu'):
    """Train face detection model"""
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    # Create checkpoints directory
    os.makedirs('face_checkpoints', exist_ok=True)

    best_map = 0.0

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs} - YOLO Face Detection Training")

        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        avg_val_loss, map_value = evaluate_model(model, val_dataloader, device)

        logging.info(f"  Training Loss: {avg_train_loss:.4f}")
        logging.info(f"  Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"  Validation mAP: {map_value:.4f}")

        is_best = map_value > best_map
        best_map = max(best_map, map_value)

        save_model_checkpoint(
            model, optimizer, epoch, avg_train_loss, avg_val_loss, map_value, best_map, is_best
        )

        lr_scheduler.step()

    logging.info(f"\nTraining completed! Best mAP: {best_map:.4f}")
    return model