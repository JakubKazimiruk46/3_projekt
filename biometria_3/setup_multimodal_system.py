# setup_multimodal_system.py
"""
Skrypt konfiguracyjny dla systemu biometrycznego multimodalnego
Instaluje wymagane biblioteki i tworzy strukturę katalogów
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Instaluje wymagane biblioteki"""
    requirements = [
        "deepface>=0.0.79",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0", 
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pillow>=10.0.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
        "kagglehub>=0.2.0"
    ]
    
    print("Instalowanie wymaganych bibliotek...")
    for req in requirements:
        print(f"Instaluję: {req}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ {req}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Błąd instalacji {req}: {e}")
            return False
    
    print("✓ Wszystkie biblioteki zainstalowane!")
    return True

def create_directory_structure():
    """Tworzy strukturę katalogów dla projektu"""
    
    directories = [
        "face_photos",
        "signature_photos", 
        "test_data/faces",
        "test_data/signatures",
        "results",
        "models"
    ]
    
    # Utwórz katalogi użytkowników dla twarzy
    for i in range(1, 21):  # 20 użytkowników dla przykładu
        directories.append(f"face_photos/user{i:02d}")
        directories.append(f"signature_photos/user{i:02d}")
        directories.append(f"test_data/faces/user{i:02d}")
        directories.append(f"test_data/signatures/user{i:02d}")
    
    print("Tworzenie struktury katalogów...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")
    
    return True

def create_readme():
    """Tworzy plik README z instrukcjami"""
    readme_content = """# System Biometryczny Multimodalny

System łączący rozpoznawanie twarzy i podpisu zgodnie z wymaganiami zadania.

## Struktura projektu

```
project/
├── face_recognition_system.py     # System rozpoznawania twarzy (DeepFace)
├── multimodal_biometric_system.py # System multimodalny (główny)
├── signature_recognition.py       # System rozpoznawania podpisów (istniejący)
├── setup_multimodal_system.py     # Skrypt instalacyjny
│
├── face_photos/                   # Zdjęcia twarzy użytkowników
│   ├── user01/
│   │   ├── photo1.jpg
│   │   ├── photo2.jpg
│   │   └── ... (minimum 5 zdjęć)
│   ├── user02/
│   └── ...
│
├── signature_photos/              # Podpisy użytkowników  
│   ├── user01/
│   │   ├── signature1.jpg
│   │   └── ... (minimum 5 podpisów)
│   └── ...
│
├── test_data/                     # Dane testowe
│   ├── faces/
│   └── signatures/
│
├── results/                       # Wyniki ewaluacji
└── models/                        # Zapisane modele
```

## Instrukcja użycia

### 1. Instalacja i konfiguracja
```bash
python setup_multimodal_system.py
```

### 2. Przygotowanie danych

**Zdjęcia twarzy:**
- Dodaj minimum 5 zdjęć każdego użytkownika do `face_photos/userXX/`
- Formaty: JPG, PNG, BMP
- Zalecane: zdjęcia z dobrym oświetleniem, twarz wyraźnie widoczna
- Rozmiar: minimum 224x224 pikseli

**Podpisy:**
- Użyj istniejących danych z `signature_recognition.py` lub
- Dodaj nowe podpisy do `signature_photos/userXX/`
- Te same użytkownicy co w zdjęciach twarzy!

### 3. Trenowanie modelu podpisów (jeśli potrzebne)
```bash
python signature_recognition.py
```

### 4. Uruchomienie systemu multimodalnego
```bash
python multimodal_biometric_system.py
```

## Algorytm systemu multimodalnego

Zgodnie z wymaganiami zadania:

1. **Rozpoznawanie dwóch cech:**
   - Twarz (DeepFace/face-recognition)
   - Podpis (istniejący system z MLP)

2. **Obliczanie pewności łącznej:**
   ```
   pewność_całość = w_face * pewność_face + w_signature * pewność_signature
   ```
   gdzie w_face + w_signature = 1

3. **Logika decyzyjna:**
   - Jeśli pewność_całość > 0.5:
     - Jeśli oba systemy zwróciły tę samą tożsamość → zwróć tożsamość
     - Jeśli różne tożsamości → "brak rozpoznania"  
   - Jeśli pewność_całość ≤ 0.5 → "brak rozpoznania"

## Parametry systemu

Możesz eksperymentować z:
- `w_face`: waga dla rozpoznawania twarzy (domyślnie 0.6)
- `w_signature`: waga dla podpisu (domyślnie 0.4)  
- `confidence_threshold`: próg pewności (domyślnie 0.5)

## Ewaluacja

System automatycznie przeprowadza ewaluację porównując:
- Skuteczność tylko twarzy
- Skuteczność tylko podpisu  
- Skuteczność systemu multimodalnego

Wyniki zapisywane są do `results/multimodal_evaluation_results.json`

## Wymagania techniczne

- Python 3.8+
- CUDA (opcjonalnie, dla przyspieszenia)
- Minimum 4GB RAM
- ~2GB miejsca na dysku (modele DeepFace)

## Rozwiązywanie problemów

**Błąd "No face found":**
- Sprawdź jakość zdjęć
- Użyj zdjęć z lepszym oświetleniem
- Spróbuj inny detector_backend w FaceRecognitionSystem

**Błąd modelu podpisów:**
- Uruchom najpierw `signature_recognition.py`
- Sprawdź czy pliki `mlp_model_multi.pkl` i `scaler_multi.pkl` istnieją

**Problemy z wydajnością:**
- Zmniejsz liczbę zdjęć na użytkownika
- Użyj mniejszy model (np. "VGG-Face" zamiast "Facenet512")
"""

    with open("README_multimodal.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✓ README_multimodal.md utworzony")

def create_example_config():
    """Tworzy przykładowy plik konfiguracyjny"""
    config_content = """# config_multimodal.py

Konfiguracja systemu multimodalnego
Dostosuj parametry według potrzeb


# Parametry systemu multimodalnego
MULTIMODAL_CONFIG = {
    # Wagi dla różnych cech biometrycznych (muszą sumować się do 1.0)
    "w_face": 0.6,        # Waga dla rozpoznawania twarzy
    "w_signature": 0.4,   # Waga dla rozpoznawania podpisu
    
    # Próg decyzyjny
    "confidence_threshold": 0.5,
    
    # Ścieżki
    "face_photos_dir": "face_photos",
    "signature_photos_dir": "signature_photos", 
    "face_database_file": "models/face_database.json",
    "signature_model_file": "models/mlp_model_multi.pkl",
    "signature_scaler_file": "models/scaler_multi.pkl",
    
    # Wyniki
    "results_dir": "results",
    "evaluation_file": "results/multimodal_evaluation.json"
}

# Parametry systemu twarzy
FACE_CONFIG = {
    "model_name": "Facenet512",     # "Facenet512", "ArcFace", "VGG-Face" 
    "detector_backend": "opencv",   # "opencv", "mtcnn", "retinaface"
    "distance_metric": "cosine",    # "cosine", "euclidean", "euclidean_l2"
}

# Eksperymentalne konfiguracje do testowania
EXPERIMENTAL_CONFIGS = [
    {"w_face": 0.7, "w_signature": 0.3, "threshold": 0.5},
    {"w_face": 0.5, "w_signature": 0.5, "threshold": 0.5},
    {"w_face": 0.6, "w_signature": 0.4, "threshold": 0.4},
    {"w_face": 0.6, "w_signature": 0.4, "threshold": 0.6},
]
"""

    with open("config_multimodal.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✓ config_multimodal.py utworzony")

def main():
    """Główna funkcja konfiguracyjna"""
    print("KONFIGURACJA SYSTEMU BIOMETRYCZNEGO MULTIMODALNEGO")
    print("=" * 55)
    
    print("\n1. Sprawdzanie wersji Python...")
    if sys.version_info < (3, 8):
        print("✗ Wymagany Python 3.8 lub nowszy!")
        return False
    print(f"✓ Python {sys.version}")
    
    print("\n2. Instalowanie bibliotek...")
    if not install_requirements():
        print("✗ Błąd instalacji bibliotek!")
        return False
    
    print("\n3. Tworzenie struktury katalogów...")
    if not create_directory_structure():
        print("✗ Błąd tworzenia katalogów!")
        return False
    
    print("\n4. Tworzenie plików konfiguracyjnych...")
    create_readme()
    create_example_config()
    
    print("\n" + "=" * 55)
    print("✓ KONFIGURACJA ZAKOŃCZONA POMYŚLNIE!")
    print("\nKolejne kroki:")
    print("1. Dodaj zdjęcia użytkowników do face_photos/userXX/")
    print("2. Dodaj podpisy do signature_photos/userXX/ (lub użyj istniejących)")
    print("3. Uruchom: python signature_recognition.py (jeśli potrzebne)")
    print("4. Uruchom: python multimodal_biometric_system.py")
    print("\nSzczegóły w pliku: README_multimodal.md")
    print("=" * 55)
    
    return True

if __name__ == "__main__":
    main()