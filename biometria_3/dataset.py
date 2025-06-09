# dataset.py
import os
import torch
from PIL import Image
from glob import glob
import logging

FORMATS = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")


class YOLOFaceDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path, class_names, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform

        # Mapowanie nazw klas na ich indeksy (YOLO uses 0-indexed, PyTorch detection uses 1-indexed)
        self.labels_mapping = {idx: idx + 1 for idx in range(len(class_names))}
        logging.info(f"YOLO Face detection classes: {class_names}")
        logging.info(f"Class mapping (YOLO -> PyTorch): {self.labels_mapping}")

        # Find all valid image-label pairs
        self.valid_samples = self._find_valid_samples()
        logging.info(
            f"Created YOLO face dataset with {len(self.valid_samples)} samples from {os.path.basename(images_path)}")

        # Show sample data
        if len(self.valid_samples) > 0:
            logging.info("Sample YOLO data entries:")
            for i in range(min(3, len(self.valid_samples))):
                img_path, label_path = self.valid_samples[i]
                logging.info(f"  Sample {i + 1}: {os.path.basename(img_path)} -> {os.path.basename(label_path)}")

    def _find_valid_samples(self):
        """Find all valid image-label pairs"""
        valid_samples = []

        # Find all image files
        image_files = []
        for ext in FORMATS:
            pattern = os.path.join(self.images_path, f"*{ext}")
            image_files.extend(glob(pattern))

        logging.info(f"Found {len(image_files)} images in {self.images_path}")

        for img_path in image_files:
            # Get corresponding label file
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_path, f"{img_name}.txt")

            if os.path.exists(label_path):
                valid_samples.append((img_path, label_path))
            else:
                logging.warning(f"No label found for {os.path.basename(img_path)}")

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
                    logging.warning(f"Invalid line in {label_path}: {line}")
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
                    logging.warning(f"Invalid box in {label_path}: {line}")
                    continue

                boxes.append([x1, y1, x2, y2])
                # Convert YOLO class index to PyTorch class index (add 1)
                labels.append(self.labels_mapping.get(class_id, 1))

        except Exception as e:
            logging.error(f"Error parsing annotation {label_path}: {e}")
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
            logging.error(f"Error loading image {img_path}: {e}")
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
    return list(zip(*batch))
