# main_integrated.py
"""
Zintegrowany system łączący YOLO face detection z systemem multimodalnym
Zachowuje strukturę projektu YOLO i dodaje funkcjonalność multimodalną
"""

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

# Import systemu multimodalnego
from multimodal_biometric_system import MultimodalBiometricSystem
from signature_recognition import main as train_signatures

# Konfiguracja logowania
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


def show_menu():
    """Wyświetla menu główne systemu"""
    print("\n" + "="*60)
    print("ZINTEGROWANY SYSTEM BIOMETRYCZNY")
    print("="*60)
    print("1. Trenowanie modelu YOLO (detekcja twarzy)")
    print("2. Detekcja twarzy z kamery (YOLO)")
    print("3. Trenowanie modeli podpisów")
    print("4. System multimodalny (rozpoznawanie tożsamości)")
    print("5. Pełny pipeline (trening + multimodalny)")
    print("6. 🔴 LIVE AUTHENTICATION z kamery")  # NOWA OPCJA
    print("0. Wyjście")
    print("="*60)


def run_yolo_training():
    """Uruchamia trenowanie modelu YOLO"""
    print("\n🚀 TRENOWANIE MODELU YOLO")
    print("-" * 40)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Sprawdź strukturę projektu YOLO
    if not check_project_structure(config):
        print("❌ Nieprawidłowa struktura projektu YOLO")
        print("Wymagana struktura:")
        print("project/")
        print("├── images/")
        print("│   ├── train/")
        print("│   └── validation/")
        print("└── labels/")
        print("    ├── train/")
        print("    └── validation/")
        return False
    
    try:
        # Tworzenie datasetów
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
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("❌ Brak danych treningowych lub walidacyjnych")
            return False
        
        # DataLoadery
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, 
            collate_fn=collate_fn, num_workers=0
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        
        # Model i trenowanie
        model = initialize_face_model(num_classes=len(config.class_names))
        trained_model = train_face_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=config.num_epochs,
            device=device
        )
        
        print("✅ Trenowanie YOLO zakończone pomyślnie!")
        return trained_model
        
    except Exception as e:
        print(f"❌ Błąd podczas trenowania YOLO: {e}")
        logging.exception("YOLO training error")
        return False


def run_yolo_webcam():
    """Uruchamia detekcję z kamery YOLO"""
    print("\n📹 DETEKCJA TWARZY Z KAMERY")
    print("-" * 40)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sprawdź czy jest wytrenowany model
    checkpoint_path = os.path.join(config.checkpoint_dir, "best_yolo_face_model.pth")
    if not os.path.exists(checkpoint_path):
        print("❌ Brak wytrenowanego modelu YOLO")
        print("Uruchom najpierw opcję 1 (Trenowanie modelu YOLO)")
        return False
    
    try:
        # Wczytaj model
        model = initialize_face_model(num_classes=len(config.class_names))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("✅ Model YOLO wczytany pomyślnie!")
        ask_for_webcam_inference(model, config, device)
        return True
        
    except Exception as e:
        print(f"❌ Błąd wczytywania modelu: {e}")
        return False


def run_signature_training():
    """Uruchamia trenowanie modeli podpisów"""
    print("\n✍️ TRENOWANIE MODELI PODPISÓW")
    print("-" * 40)
    
    try:
        # Uruchom trenowanie podpisów
        train_signatures()
        print("✅ Trenowanie modeli podpisów zakończone!")
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas trenowania podpisów: {e}")
        logging.exception("Signature training error")
        return False


def run_multimodal_system():
    """Uruchamia system multimodalny"""
    print("\n🔄 SYSTEM MULTIMODALNY")
    print("-" * 40)
    
    # Sprawdź czy istnieją wymagane modele
    required_files = [
        "mlp_model_multi.pkl",
        "scaler_multi.pkl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Brak wymaganych modeli:")
        for f in missing_files:
            print(f"  - {f}")
        print("Uruchom najpierw opcję 3 (Trenowanie modeli podpisów)")
        return False
    
    # Sprawdź czy istnieją dane twarzy
    if not os.path.exists("face_photos"):
        print("❌ Brak katalogu face_photos")
        print("Utwórz katalog face_photos/user01/ i dodaj zdjęcia twarzy")
        return False
    
    try:
        # Konfiguracja systemu multimodalnego
        multimodal_system = MultimodalBiometricSystem(
            w_face=0.6,
            w_signature=0.4,
            confidence_threshold=0.5
        )
        
        # Konfiguracja systemu twarzy
        print("Konfiguracja systemu rozpoznawania twarzy...")
        face_success = multimodal_system.setup_face_system(
            "face_photos", 
            "face_database.json"
        )
        
        if not face_success:
            print("❌ Błąd konfiguracji systemu twarzy")
            return False
        
        # Konfiguracja systemu podpisów
        print("Konfiguracja systemu rozpoznawania podpisów...")
        signature_success = multimodal_system.setup_signature_system()
        
        if not signature_success:
            print("❌ Błąd konfiguracji systemu podpisów")
            return False
        
        print("✅ Systemy skonfigurowane pomyślnie!")
        
        # Ewaluacja
        print("\nCzy uruchomić ewaluację systemu? (y/n): ", end="")
        if input().lower() in ['y', 'yes', 'tak']:
            
            # Sprawdź czy istnieją dane testowe
            if os.path.exists("test_data"):
                from multimodal_biometric_system import create_test_dataset_from_test_data
                test_data = create_test_dataset_from_test_data("test_data", "user01", photos_per_user=5)
            else:
                # Fallback - użyj ostatnich zdjęć z face_photos
                from multimodal_biometric_system import create_test_dataset_single_user
                kaggle_path = "C:/Users/Jakub/.cache/kagglehub/datasets/robinreni/signature-verification-dataset/versions/2/sign_data/train"
                test_data = create_test_dataset_single_user("face_photos", kaggle_path, "user01", 2)
            
            if test_data:
                print(f"Uruchamianie ewaluacji z {len(test_data)} próbkami...")
                evaluation_results = multimodal_system.evaluate_system(test_data)
                multimodal_system.print_evaluation_report(evaluation_results)
                multimodal_system.save_evaluation_results(evaluation_results, "multimodal_evaluation_results.json")
                print("✅ Wyniki zapisane do: multimodal_evaluation_results.json")
            else:
                print("❌ Brak danych testowych do ewaluacji")
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd systemu multimodalnego: {e}")
        logging.exception("Multimodal system error")
        return False


def run_full_pipeline():
    """Uruchamia pełny pipeline: trening podpisów + system multimodalny"""
    print("\n🚀 PEŁNY PIPELINE")
    print("-" * 40)
    
    print("Krok 1: Trenowanie modeli podpisów...")
    if not run_signature_training():
        return False
    
    print("\nKrok 2: System multimodalny...")
    if not run_multimodal_system():
        return False
    
    print("✅ Pełny pipeline zakończony pomyślnie!")
    return True


def main():
    """Główna funkcja z menu"""
    
    while True:
        show_menu()
        
        try:
            choice = input("\nWybierz opcję (0-6): ").strip()
            
            if choice == "0":
                print("👋 Do widzenia!")
                break
                
            elif choice == "1":
                run_yolo_training()
                
            elif choice == "2":
                run_yolo_webcam()
                
            elif choice == "3":
                run_signature_training()
                
            elif choice == "4":
                run_multimodal_system()
                
            elif choice == "5":
                run_full_pipeline()
                
            elif choice == "6":
                # NOWA OPCJA - Live Authentication
                from live_multimodal_webcam import run_live_authentication
                run_live_authentication()
                
            else:
                print("❌ Nieprawidłowa opcja. Wybierz 0-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Przerwano przez użytkownika")
            break
        except Exception as e:
            print(f"❌ Nieoczekiwany błąd: {e}")
            logging.exception("Unexpected error in main menu")
        
        input("\nNaciśnij Enter aby kontynuować...")


if __name__ == "__main__":
    main()