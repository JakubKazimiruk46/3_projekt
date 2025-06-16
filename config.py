# config.py
import os
from dataclasses import dataclass
from torchvision import transforms as T


# Klasa konfiguracji projektu – wszystkie ścieżki i parametry zebrane w jednym miejscu
@dataclass
class Config:
    enable_training: bool = True
    project_root: str = 'project'  # Główny katalog projektu z danymi
    checkpoint_dir: str = 'face_checkpoints'
    batch_size: int = 16
    num_epochs: int = 30
    num_visualizations: int = 5
    face_confidence_threshold: float = 0.5  # Próg ufności detekcji twarzy (dla webcam)

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

    @property
    def class_names(self):
        return ['face']

    # Transformacje obrazu – konwersja do tensora i normalizacja zgodna z ImageNet
    @property
    def transform(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
