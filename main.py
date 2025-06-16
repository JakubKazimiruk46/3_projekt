import logging
import multiprocessing
import os
import warnings
import torch
from config import Config
from dataset import YOLOFaceDataset, collate_fn
from training import (
    initialize_face_model,
    train_face_model,
)
from webcam import ask_for_webcam_inference
from utils import check_project_structure

# Konfiguracja logowania – zapis do pliku i logi w konsoli
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings('ignore')
multiprocessing.freeze_support()


def create_yolo_datasets(config: Config):
    """
    Tworzy zbiory treningowe i walidacyjne z plików w formacie YOLO.
    """
    # Sprawdź, czy wszystkie wymagane ścieżki istnieją
    required_paths = [
        config.train_images_path,
        config.train_labels_path,
        config.val_images_path,
        config.val_labels_path
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required directory not found: {path}")

    logging.info("Creating YOLO datasets from project structure:")
    logging.info(f"  Train images: {config.train_images_path}")
    logging.info(f"  Train labels: {config.train_labels_path}")
    logging.info(f"  Val images: {config.val_images_path}")
    logging.info(f"  Val labels: {config.val_labels_path}")

    # Utwórz datasety
    train_dataset = YOLOFaceDataset(
        images_path=config.train_images_path,
        labels_path=config.train_labels_path,
        class_names=config.class_names,
        transform=config.transform
    )

    val_dataset = YOLOFaceDataset(
        images_path=config.val_images_path,
        labels_path=config.val_labels_path,
        class_names=config.class_names,
        transform=config.transform
    )

    return train_dataset, val_dataset


def create_dataloaders(train_data, val_data, batch_size=16):
    """
    Tworzy DataLoadery dla zbiorów treningowych i walidacyjnych.
    """
    if len(train_data) == 0:
        logging.error("Training dataset is empty!")
        raise ValueError("Training dataset is empty!")
    if len(val_data) == 0:
        logging.error("Validation dataset is empty!")
        raise ValueError("Validation dataset is empty!")

    logging.info(f"Creating dataloaders with batch size: {batch_size}")

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


def train_or_load_model(config: Config, device: torch.device):
    """
    Główna funkcja decydująca czy trenować model od zera czy wczytać checkpoint.
    """
    model = initialize_face_model(num_classes=len(config.class_names))

    if config.enable_training:
        logging.info("Starting YOLO face detection training...")
        try:
            train_dataset, val_dataset = create_yolo_datasets(config)

            if len(train_dataset) == 0 or len(val_dataset) == 0:
                logging.error("Training or validation dataset is empty!")
                return None

            train_loader, val_loader = create_dataloaders(
                train_dataset, val_dataset, batch_size=config.batch_size
            )

            model = train_face_model(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                num_epochs=config.num_epochs,
                device=device
            )
            logging.info("YOLO face detection training completed successfully!")
            return model

        except Exception as e:
            logging.exception(f"An error occurred during training: {e}")
            return None
    else:
        # Pomijamy trening – próbujemy wczytać gotowy model z pliku
        logging.info("Skipping training phase, loading saved checkpoint...")
        checkpoint_path = os.path.join(config.checkpoint_dir, "best_yolo_face_model.pth")
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded model with mAP: {checkpoint.get('best_map', 0):.4f}")
            return model
        else:
            logging.warning(f"No checkpoint found at {checkpoint_path}")
            return None


def main():
    # Utwórz konfigurację oraz wybierz urządzenie (GPU/CPU)
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"YOLO Face detection classes: {config.class_names}")

    # Sprawdź strukturę projektu (czy wszystkie katalogi istnieją)
    if not check_project_structure(config):
        return

    # Trening lub wczytanie modelu
    model = train_or_load_model(config, device)

    # Po zakończonym treningu lub załadowaniu modelu, zapytaj użytkownika o detekcję z kamery
    if model is not None:
        ask_for_webcam_inference(model, config, device)


if __name__ == "__main__":
    main()
