import os
import logging
from config import Config


def check_project_structure(config: Config) -> bool:
    """Sprawdza czy projekt ma wymagany układ katalogów"""
    logging.info(f"Checking project structure in: {config.project_root}")
    required_paths = [
        config.train_images_path,
        config.train_labels_path,
        config.val_images_path,
        config.val_labels_path
    ]

    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        logging.error("Missing required directories:")
        for path in missing_paths:
            logging.error(f"  - {path}")
        logging.error("Please ensure you have the following structure:\n"
                      "project/\n"
                      "├── images/\n"
                      "│   ├── train/\n"
                      "│   └── validation/\n"
                      "└── labels/\n"
                      "    ├── train/\n"
                      "    └── validation/")
        return False
    return True
