import json
import os
import random
import warnings
from glob import glob
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm
import multiprocessing
import cv2

IOU_THRESHOLDS = [0.3]

warnings.filterwarnings('ignore')
multiprocessing.freeze_support()

# Face detection classes - YOLO format uses class indices
class_names = ['face']  # Class 0 = face
# If you have multiple classes, add them here: ['face', 'person', 'hand', etc.]

FORMATS = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

# ImageNet normalization for pretrained models
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@dataclass
class Config:
    enable_training: bool = True
    project_root: str = 'project'  # Root project directory
    checkpoint_dir: str = 'face_checkpoints'
    batch_size: int = 16
    num_epochs: int = 30
    num_visualizations: int = 5
    face_confidence_threshold: float = 0.5

    @property
    def train_images_path(self):
        return os.path.join(self.project_root, 'images', 'train')
    
    @property
    def train_labels_path(self):
        return os.path.join(self.project_root, 'labels', 'train')
    
    @property
    def val_images_path(self):
        return os.path.join(self.project_root, 'images', 'validation')
    
    @property
    def val_labels_path(self):
        return os.path.join(self.project_root, 'labels', 'validation')


class YOLOFaceDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        
        # Mapowanie nazw klas na ich indeksy (YOLO uses 0-indexed, PyTorch detection uses 1-indexed)
        self.labels_mapping = {idx: idx + 1 for idx in range(len(class_names))}
        print(f"YOLO Face detection classes: {class_names}")
        print(f"Class mapping (YOLO -> PyTorch): {self.labels_mapping}")

        # Find all valid image-label pairs
        self.valid_samples = self._find_valid_samples()
        print(f"Created YOLO face dataset with {len(self.valid_samples)} samples from {os.path.basename(images_path)}")

        # Show sample data
        if len(self.valid_samples) > 0:
            print("Sample YOLO data entries:")
            for i in range(min(3, len(self.valid_samples))):
                img_path, label_path = self.valid_samples[i]
                print(f"  Sample {i + 1}: {os.path.basename(img_path)} -> {os.path.basename(label_path)}")

    def _find_valid_samples(self):
        """Find all valid image-label pairs"""
        valid_samples = []
        
        # Find all image files
        image_files = []
        for ext in FORMATS:
            pattern = os.path.join(self.images_path, f"*{ext}")
            image_files.extend(glob(pattern))
        
        print(f"Found {len(image_files)} images in {self.images_path}")
        
        for img_path in image_files:
            # Get corresponding label file
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_path, f"{img_name}.txt")
            
            if os.path.exists(label_path):
                valid_samples.append((img_path, label_path))
            else:
                print(f"Warning: No label found for {os.path.basename(img_path)}")
        
        return valid_samples

    def _parse_yolo_annotation(self, label_path):
        """Parse YOLO format annotation file"""
        boxes = []
        labels = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # YOLO format: class_id center_x center_y width height
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: Invalid line in {label_path}: {line}")
                    continue
                
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert YOLO format (center_x, center_y, width, height) to 
                # PyTorch format (x1, y1, x2, y2) in normalized coordinates
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2
                
                # Ensure coordinates are within [0, 1]
                x1 = max(0, min(1, x1))
                y1 = max(0, min(1, y1))
                x2 = max(0, min(1, x2))
                y2 = max(0, min(1, y2))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid box in {label_path}: {line}")
                    continue
                
                boxes.append([x1, y1, x2, y2])
                # Convert YOLO class index to PyTorch class index (add 1)
                labels.append(self.labels_mapping.get(class_id, 1))
        
        except Exception as e:
            print(f"Error parsing annotation {label_path}: {e}")
            return [], []
        
        return boxes, labels

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        # Get image and label paths
        img_path, label_path = self.valid_samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        width, height = image.size

        # Parse YOLO annotation
        boxes_norm, labels = self._parse_yolo_annotation(label_path)
        
        # Convert normalized coordinates to pixel coordinates
        boxes_pixel = []
        for box in boxes_norm:
            x1, y1, x2, y2 = box
            x1_pixel = x1 * width
            y1_pixel = y1 * height
            x2_pixel = x2 * width
            y2_pixel = y2 * height
            boxes_pixel.append([x1_pixel, y1_pixel, x2_pixel, y2_pixel])

        # Convert to tensors
        if len(boxes_pixel) > 0:
            boxes = torch.tensor(boxes_pixel, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # If no annotations, create empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Prepare target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
        }

        return image, target


def collate_fn(batch):
    """Custom collate function for variable-sized batches"""
    return list(zip(*batch))


def create_yolo_datasets(config: Config):
    """
    Create training and validation datasets from YOLO format data
    """
    # Check if all required paths exist
    required_paths = [
        config.train_images_path,
        config.train_labels_path,
        config.val_images_path,
        config.val_labels_path
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required directory not found: {path}")

    print(f"Creating YOLO datasets from project structure:")
    print(f"  Train images: {config.train_images_path}")
    print(f"  Train labels: {config.train_labels_path}")
    print(f"  Val images: {config.val_images_path}")
    print(f"  Val labels: {config.val_labels_path}")

    # Create datasets
    train_dataset = YOLOFaceDataset(
        images_path=config.train_images_path,
        labels_path=config.train_labels_path,
        transform=transform
    )

    val_dataset = YOLOFaceDataset(
        images_path=config.val_images_path,
        labels_path=config.val_labels_path,
        transform=transform
    )

    return train_dataset, val_dataset


def create_dataloaders(train_data, val_data, batch_size=16):
    """Create data loaders for training and validation datasets"""
    if len(train_data) == 0:
        raise ValueError("Training dataset is empty!")
    if len(val_data) == 0:
        raise ValueError("Validation dataset is empty!")

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_dataloader, val_dataloader


def initialize_face_model(num_classes):
    """Initialize face detection model"""
    # SSDLite with MobileNetV3 - good for real-time face detection
    model = models.detection.ssdlite320_mobilenet_v3_large(
        pretrained=False,
        num_classes=num_classes + 1  # +1 for background class
    )

    return model


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

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        print(f"\nEpoch {epoch + 1}/{num_epochs} - YOLO Face Detection Training")

        running_loss = 0.0
        num_batches = 0

        for i, (images, targets) in enumerate(tqdm(train_dataloader, desc="Training")):
            # Skip batches with no valid targets
            valid_indices = [j for j, t in enumerate(targets) if len(t['boxes']) > 0]
            if len(valid_indices) == 0:
                continue

            # Filter valid images and targets
            valid_images = [images[j] for j in valid_indices]
            valid_targets = [targets[j] for j in valid_indices]
            
            # Move to device
            images_device = [img.to(device) for img in valid_images]
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in valid_targets]

            # Zero gradients
            optimizer.zero_grad()

            try:
                # Forward pass
                loss_dict = model(images_device, targets_device)
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                num_batches += 1

                # Print loss every 20 batches
                if i % 20 == 19 and num_batches > 0:
                    avg_loss = running_loss / num_batches
                    print(f"  Batch {i + 1}, Avg Loss: {avg_loss:.4f}")

            except Exception as e:
                print(f"Error in training batch {i}: {e}")
                continue

        # Update learning rate
        lr_scheduler.step()

        # Validation phase
        print("Evaluating on validation set...")
        model.eval()

        metric = MeanAveragePrecision(iou_thresholds=IOU_THRESHOLDS)
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(val_dataloader, desc="Validation"):
                # Skip empty batches
                valid_indices = [j for j, t in enumerate(targets) if len(t['boxes']) > 0]
                if len(valid_indices) == 0:
                    continue

                valid_images = [images[j] for j in valid_indices]
                valid_targets = [targets[j] for j in valid_indices]
                
                images_device = [img.to(device) for img in valid_images]
                targets_device = [{k: v.to(device) for k, v in t.items()} for t in valid_targets]
                
                try:
                    # Get predictions
                    predictions = model(images_device)
                    
                    # Calculate validation loss
                    model.train()  # Temporarily switch to train mode for loss calculation
                    loss_dict = model(images_device, targets_device)
                    val_loss += sum(loss for loss in loss_dict.values()).item()
                    val_batches += 1
                    model.eval()  # Switch back to eval mode
                    
                    # Convert to CPU for metrics
                    predictions_cpu = [{k: v.cpu() for k, v in p.items()} for p in predictions]
                    targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in valid_targets]

                    metric.update(predictions_cpu, targets_cpu)
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        # Compute metrics
        try:
            metrics = metric.compute()
            map_value = metrics['map'].item()
        except:
            map_value = 0.0

        avg_train_loss = running_loss / max(num_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)

        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation mAP: {map_value:.4f}")

        # Save model
        is_best = map_value > best_map
        if is_best:
            best_map = map_value
            checkpoint_path = f"face_checkpoints/best_yolo_face_model.pth"
            print(f"  New best model! mAP: {map_value:.4f}")
        else:
            checkpoint_path = f"face_checkpoints/yolo_face_model_epoch_{epoch + 1}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'map': map_value,
            'best_map': best_map,
            'class_names': class_names
        }, checkpoint_path)
        print(f"  Model saved to {checkpoint_path}")

    print(f"\nTraining completed! Best mAP: {best_map:.4f}")
    return model


def webcam_face_detection(model, device, confidence_threshold=0.5, enable_fps_display=True, mirror=True):
    """Live webcam face detection"""
    import time
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    model.eval()
    fps_history = []
    
    print("Live YOLO Face Detection Started")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    paused = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam")
            break
        
        if mirror:
            frame = cv2.flip(frame, 1)
        
        display_frame = frame.copy()
        
        if not paused:
            start_time = time.time()
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # Perform detection
            with torch.no_grad():
                predictions = model(input_tensor)
            
            # Process predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            # Draw bounding boxes
            faces_detected = 0
            
            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    
                    faces_detected += 1
                    color = (0, 255, 0)  # Green for faces
                    
                    # Draw rectangle and label
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"Face: {score:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display status
            status_text = f"{faces_detected} Face(s) Detected" if faces_detected > 0 else "No Faces Detected"
            status_color = (0, 255, 0) if faces_detected > 0 else (0, 0, 255)
            cv2.putText(display_frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # FPS display
            if enable_fps_display:
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)
                
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "PAUSED (Press 'p' to resume)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('YOLO Face Detection', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    print("YOLO face detection stopped")


def main():
    config = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"YOLO Face detection classes: {class_names}")

    # Check project structure
    print(f"\nChecking project structure in: {config.project_root}")
    required_paths = [
        config.train_images_path,
        config.train_labels_path,
        config.val_images_path,
        config.val_labels_path
    ]
    
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        print("Error: Missing required directories:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nPlease ensure you have the following structure:")
        print("project/")
        print("├── images/")
        print("│   ├── train/")
        print("│   └── validation/")
        print("└── labels/")
        print("    ├── train/")
        print("    └── validation/")
        return

    # Initialize model
    model = initialize_face_model(num_classes=len(class_names))

    if config.enable_training:
        print("\nStarting YOLO face detection training...")
        try:
            train_dataset, val_dataset = create_yolo_datasets(config)

            if len(train_dataset) == 0:
                print("Error: Training dataset is empty!")
                return
            if len(val_dataset) == 0:
                print("Error: Validation dataset is empty!")
                return

            train_dataloader, val_dataloader = create_dataloaders(
                train_dataset, val_dataset, batch_size=config.batch_size
            )

            model = train_face_model(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=config.num_epochs,
                device=device
            )
            print("YOLO face detection training completed successfully!")

        except Exception as e:
            print(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("Skipping training phase, loading saved checkpoint...")
        # Load best checkpoint if available
        best_checkpoint = os.path.join(config.checkpoint_dir, "best_yolo_face_model.pth")
        if os.path.exists(best_checkpoint):
            print(f"Loading checkpoint: {best_checkpoint}")
            checkpoint = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model with mAP: {checkpoint.get('best_map', 0):.4f}")
        else:
            print(f"No checkpoint found at {best_checkpoint}")

    # Ask for live detection
    print("\nDo you want to start live webcam face detection? (y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        print("Starting live YOLO face detection...")
        webcam_face_detection(
            model=model,
            device=device,
            confidence_threshold=config.face_confidence_threshold,
            enable_fps_display=True,
            mirror=True
        )


if __name__ == "__main__":
    main()