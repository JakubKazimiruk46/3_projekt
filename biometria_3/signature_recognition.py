# signature_recognition_custom.py
"""
Trening modelu podpisów na Twoich własnych danych zamiast Kaggle
"""

import os
import cv2
import numpy as np
from PIL import Image
import time
from models import train_and_evaluate
from visualization import visualize_pca, visualize_tsne


def preprocess_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")  # konwersja na skalę szarości
    except Exception:
        return None
    img = np.array(img)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary  # odwrócenie kolorów (podpis na czarno)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        binary = binary[y:y + h, x:x + w]  # segmentacja podpisu

    resized = cv2.resize(binary, (256, 128))  # normalizacja rozmiaru

    # Ścienianie - obsługa różnych metod
    try:
        if hasattr(cv2, 'ximgproc'):
            thinned = cv2.ximgproc.thinning(resized)
        else:
            # Fallback - proste ścienianie
            kernel = np.ones((3,3), np.uint8)
            thinned = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel)
            thinned = cv2.erode(thinned, kernel, iterations=1)
    except:
        thinned = resized

    return thinned


def extract_features(image):
    if image is None:
        return [0] * 20  # zabezpieczenie

    inverted = 255 - image  # potrzebne do analizy konturów

    features = []
    h, w = image.shape

    # cecha 1: liczba czarnych pikseli
    black_pixels = np.sum(image == 0)
    # cecha 2: proporcja boków (szer./wys.)
    aspect_ratio = w / h
    features.extend([black_pixels, aspect_ratio])

    # cecha 3: współczynnik wypełnienia
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        area = cv2.contourArea(contours[0])
        x, y, w_box, h_box = cv2.boundingRect(contours[0])
        bbox_area = w_box * h_box
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
    else:
        fill_ratio = 0
    features.append(fill_ratio)

    # cechy 4–10: Hu Moments
    moments = cv2.moments(image)
    hu = cv2.HuMoments(moments).flatten()
    features.extend(hu)

    # cecha 11: liczba konturów
    contours_all, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    features.append(len(contours_all))

    # cechy 12–15: liczba czarnych pikseli w podregionach
    left = image[:, :w // 2]
    right = image[:, w // 2:]
    features.extend([np.sum(left == 0), np.sum(right == 0)])
    top = image[:h // 2, :]
    bottom = image[h // 2:, :]
    features.extend([np.sum(top == 0), np.sum(bottom == 0)])

    return features


def load_custom_data(dataset_path):
    """Wczytuje dane z Twoich katalogów signature_photos"""
    features = []
    labels = []
    binary_labels = []
    
    print(f"Loading custom data from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return np.array([]), np.array([]), np.array([])
    
    processed_count = 0
    failed_count = 0
    
    for user_folder in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_folder)
        if os.path.isdir(user_path):
            print(f"Processing user: {user_folder}")
            user_files = 0
            
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                valid_ext = (".png", ".jpg", ".jpeg", ".bmp")
                if not img_name.lower().endswith(valid_ext):
                    continue
                
                try:
                    img = preprocess_image(img_path)
                    if img is not None:
                        feat = extract_features(img)
                        features.append(feat)
                        labels.append(user_folder)
                        # Wszystkie podpisy w signature_photos to genuine
                        binary_labels.append("genuine")
                        processed_count += 1
                        user_files += 1
                    else:
                        failed_count += 1
                        print(f"  Failed to process: {img_name}")
                except Exception as e:
                    failed_count += 1
                    print(f"  Error processing {img_name}: {e}")
            
            print(f"  Processed {user_files} files for {user_folder}")
    
    print(f"Total processed: {processed_count}, failed: {failed_count}")
    
    if processed_count == 0:
        print("ERROR: No files were successfully processed!")
        return np.array([]), np.array([]), np.array([])
    
    return np.array(features), np.array(labels), np.array(binary_labels)


def main():
    start = time.time()

    # ZMIANA: Użyj Twoich katalogów zamiast Kaggle
    custom_signatures_dir = "signature_photos"
    
    print(f"Custom signatures directory: {custom_signatures_dir}")

    # Sprawdź czy katalog istnieje
    if not os.path.exists(custom_signatures_dir):
        print(f"ERROR: Custom signatures directory not found: {custom_signatures_dir}")
        print("Utwórz katalog signature_photos/user01/ i dodaj swoje podpisy!")
        return

    # Wczytanie Twoich danych
    print("Loading custom signature data...")
    features, labels, binary_labels = load_custom_data(custom_signatures_dir)
    
    if len(features) == 0:
        print("ERROR: No signature data loaded!")
        print("Dodaj pliki podpisów do signature_photos/user01/")
        return

    print(f"Features shape: {features.shape}")
    print(f"Labels unique: {np.unique(labels)}")
    print(f"Binary labels unique: {np.unique(binary_labels)}")

    # Jeśli masz tylko 1 użytkownika, dodaj sztuczne dane do treningu
    if len(np.unique(labels)) == 1:
        print("⚠️ Tylko 1 użytkownik - dodaję sztuczne dane dla treningu...")
        
        # Duplikuj dane jako "fake_user" dla celów treningu
        fake_features = features.copy()
        fake_labels = ["fake_user"] * len(features)
        fake_binary = ["forgery"] * len(features)
        
        # Połącz dane
        features = np.vstack([features, fake_features])
        labels = np.concatenate([labels, fake_labels])
        binary_labels = np.concatenate([binary_labels, fake_binary])
        
        print(f"Rozszerzone dane - users: {np.unique(labels)}")

    # Przygotowanie etykiet do klasyfikacji
    labels_dict = {
        "multi": labels,
        "binary": binary_labels
    }

    # Trening i ewaluacja modeli
    for key in labels_dict:
        print(f"\nTraining models for {key} classification...")
        train_and_evaluate(features, labels_dict[key], model_name=key)

    # Wizualizacje
    print("\nGenerating visualizations...")
    try:
        visualize_pca(features, binary_labels)
        visualize_tsne(features, binary_labels)
    except Exception as e:
        print(f"Visualization error: {e}")

    print(f"\nCzas działania: {time.time() - start:.2f} s")
    print("\n✅ Modele wytrenowane na Twoich podpisach!")
    print("Pliki wygenerowane:")
    print("  - mlp_model_multi.pkl (model dla user01)")
    print("  - scaler_multi.pkl (normalizacja)")


if __name__ == "__main__":
    main()