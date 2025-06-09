import os
import cv2
import numpy as np
import kagglehub
from PIL import Image
import time
from models import train_and_evaluate
from visualization import visualize_pca, visualize_tsne


path = kagglehub.dataset_download("robinreni/signature-verification-dataset")

print("Path to dataset files:", path)


# Punkt b: Własny algorytm wstępnego przetwarzania (segmentacja, binaryzacja, ścienianie)
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

    # Ścienianie (thinning)
    thinned = cv2.ximgproc.thinning(resized)

    return thinned


# Punkt c: Własna procedura ekstrakcji cech
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

    # cecha 3: współczynnik wypełnienia (powierzchnia podpisu względem bounding boxa)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        area = cv2.contourArea(contours[0])
        x, y, w_box, h_box = cv2.boundingRect(contours[0])
        bbox_area = w_box * h_box
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
    else:
        fill_ratio = 0
    features.append(fill_ratio)

    # cechy 4–10: Hu Moments – opis kształtu podpisu
    moments = cv2.moments(image)
    hu = cv2.HuMoments(moments).flatten()
    features.extend(hu)

    # cecha 11: liczba konturów (stopień złożoności podpisu)
    contours_all, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    features.append(len(contours_all))

    # cechy 12–15: liczba czarnych pikseli w podregionach
    left = image[:, :w // 2]
    right = image[:, w // 2:]
    features.extend([np.sum(left == 0), np.sum(right == 0)])
    top = image[:h // 2, :]
    bottom = image[h // 2:, :]
    features.extend([np.sum(top == 0), np.sum(bottom == 0)])

    # Punkt d: Opracowanie wektora cech
    return features


# Punkt a: Wczytanie zbioru podpisów (min. 10 użytkowników × 5 podpisów)
def load_data(dataset_path):
    features = []
    labels = []
    binary_labels = []
    for user_folder in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_folder)
        if os.path.isdir(user_path):
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                valid_ext = (".png", ".jpg", ".jpeg", ".bmp")
                if not img_name.lower().endswith(valid_ext):
                    continue
                img = preprocess_image(img_path)
                feat = extract_features(img)
                features.append(feat)
                labels.append(user_folder)
                binary_labels.append("forgery" if "forg" in user_folder.lower() else "genuine")
    return np.array(features), np.array(labels), np.array(binary_labels)


def main():
    start = time.time()

    train_dir = os.path.join(path, "sign_data", "train")
    test_dir = os.path.join(path, "sign_data", "test")

    # Wczytanie danych treningowych i testowych (Punkt a)
    # Zwracane są cechy, etykiety szczegółowe (dla każdego użytkownika) i binarne (genuine/forgery)
    train_feat, train_labels, train_binary = load_data(train_dir)
    test_feat, test_labels, test_binary = load_data(test_dir)

    # Połączenie danych treningowych i testowych w jeden zestaw (cechy)
    features = np.vstack((train_feat, test_feat))

    # Przygotowanie dwóch rodzajów etykiet do klasyfikacji:
    # - multi: klasyfikacja konkretnego użytkownika (wieloklasowa)
    # - binary: klasyfikacja genuiność/fałszerstwo (dwuklasowa)
    labels = {
        "multi": np.concatenate((train_labels, test_labels)),
        "binary": np.concatenate((train_binary, test_binary))
    }

    # Punkt e: Trening i ewaluacja dwóch modeli dla każdej wersji etykiet
    for key in labels:
        train_and_evaluate(features, labels[key], model_name=key)

    # Dodatkowa analiza – redukcja wymiarowości i wizualizacja przestrzeni cech (PCA, t-SNE)
    # Ułatwia ocenę separowalności klas i jakości cech (dodatkowy punkt)
    visualize_pca(features, labels["binary"])
    visualize_tsne(features, labels["binary"])

    print(f"\nCzas działania: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
