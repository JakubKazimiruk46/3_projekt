import os
import cv2
import numpy as np
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle

class FaceRecognitionSystem:
    def __init__(self):
        self.face_encodings = []
        self.face_labels = []
        self.scaler = StandardScaler()
        self.knn_model = None
        self.mlp_model = None
        
    def extract_face_features(self, image_path):
        """Ekstrakcja cech twarzy z obrazu"""
        try:
            # Wczytaj obraz
            image = face_recognition.load_image_file(image_path)
            
            # Znajdź lokalizacje twarzy
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) == 0:
                print(f"Nie znaleziono twarzy w: {image_path}")
                return None
                
            # Ekstraktuj cechy (encoding) pierwszej twarzy
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) > 0:
                return face_encodings[0]  # 128-wymiarowy wektor cech
            else:
                return None
                
        except Exception as e:
            print(f"Błąd podczas przetwarzania {image_path}: {e}")
            return None
    
    def load_face_dataset(self, dataset_path):
        """Wczytaj zbiór danych twarzy"""
        print("Ładowanie zbioru danych twarzy...")
        
        for user_folder in os.listdir(dataset_path):
            user_path = os.path.join(dataset_path, user_folder)
            
            if not os.path.isdir(user_path):
                continue
                
            print(f"Przetwarzanie użytkownika: {user_folder}")
            
            for image_file in os.listdir(user_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(user_path, image_file)
                    
                    # Ekstraktuj cechy twarzy
                    face_encoding = self.extract_face_features(image_path)
                    
                    if face_encoding is not None:
                        self.face_encodings.append(face_encoding)
                        self.face_labels.append(user_folder)
                        
        print(f"Załadowano {len(self.face_encodings)} obrazów twarzy")
        return np.array(self.face_encodings), np.array(self.face_labels)
    
    def train_face_models(self, features, labels):
        """Trenuj modele rozpoznawania twarzy"""
        if len(features) == 0:
            print("Brak danych do treningu!")
            return
            
        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Normalizacja cech
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model k-NN
        self.knn_model = KNeighborsClassifier(n_neighbors=3)
        self.knn_model.fit(X_train_scaled, y_train)
        knn_pred = self.knn_model.predict(X_test_scaled)
        knn_acc = accuracy_score(y_test, knn_pred) * 100
        
        # Model MLP
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64), 
            max_iter=2000, 
            random_state=42,
            early_stopping=True
        )
        self.mlp_model.fit(X_train_scaled, y_train)
        mlp_pred = self.mlp_model.predict(X_test_scaled)
        mlp_acc = accuracy_score(y_test, mlp_pred) * 100
        
        print(f"\n[TWARZE] Dokładność k-NN: {knn_acc:.2f}%")
        print(f"[TWARZE] Dokładność MLP: {mlp_acc:.2f}%")
        print(f"\nRaport klasyfikacji MLP - Twarze:")
        print(classification_report(y_test, mlp_pred, zero_division=0))
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_models(self, prefix="face"):
        """Zapisz wytrenowane modele"""
        joblib.dump(self.knn_model, f"{prefix}_knn_model.pkl")
        joblib.dump(self.mlp_model, f"{prefix}_mlp_model.pkl")
        joblib.dump(self.scaler, f"{prefix}_scaler.pkl")
        
        # Zapisz encodings i etykiety
        with open(f"{prefix}_data.pkl", 'wb') as f:
            pickle.dump({
                'encodings': self.face_encodings,
                'labels': self.face_labels
            }, f)
        
        print(f"Zapisano modele twarzy z prefiksem: {prefix}")
    
    def load_models(self, prefix="face"):
        """Wczytaj zapisane modele"""
        try:
            self.knn_model = joblib.load(f"{prefix}_knn_model.pkl")
            self.mlp_model = joblib.load(f"{prefix}_mlp_model.pkl")
            self.scaler = joblib.load(f"{prefix}_scaler.pkl")
            
            with open(f"{prefix}_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.face_encodings = data['encodings']
                self.face_labels = data['labels']
            
            print(f"Wczytano modele twarzy z prefiksem: {prefix}")
            return True
        except Exception as e:
            print(f"Błąd podczas wczytywania modeli: {e}")
            return False
    
    def predict_face(self, image_path, use_mlp=True):
        """Rozpoznaj twarz na obrazie"""
        face_encoding = self.extract_face_features(image_path)
        
        if face_encoding is None:
            return None, 0.0
            
        # Normalizuj cechy
        face_scaled = self.scaler.transform([face_encoding])
        
        # Przewidywanie
        if use_mlp and self.mlp_model is not None:
            prediction = self.mlp_model.predict(face_scaled)[0]
            probabilities = self.mlp_model.predict_proba(face_scaled)[0]
            confidence = np.max(probabilities)
        elif self.knn_model is not None:
            prediction = self.knn_model.predict(face_scaled)[0]
            # Dla k-NN oblicz confidence na podstawie odległości
            distances, indices = self.knn_model.kneighbors(face_scaled)
            confidence = 1.0 / (1.0 + np.mean(distances))
        else:
            return None, 0.0
            
        return prediction, confidence

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja systemu
    face_system = FaceRecognitionSystem()
    
    # Ścieżka do zbioru danych (struktura: dataset/user1/, dataset/user2/, ...)
    dataset_path = "face_dataset"
    
    # Wczytaj i przetrenuj modele
    features, labels = face_system.load_face_dataset(dataset_path)
    
    if len(features) > 0:
        face_system.train_face_models(features, labels)
        face_system.save_models("face")
        
        # Test rozpoznawania
        test_image = "test_face.jpg"  # ścieżka do testowego obrazu
        if os.path.exists(test_image):
            prediction, confidence = face_system.predict_face(test_image)
            print(f"\nRozpoznana osoba: {prediction} (pewność: {confidence:.2f})")
    else:
        print("Nie udało się wczytać danych!")