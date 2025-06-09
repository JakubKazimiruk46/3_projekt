# face_recognition_system_simple.py
"""
Prostszy system rozpoznawania twarzy używający face-recognition
Działa z Python 3.13 bez problemów z TensorFlow
"""

import face_recognition
import numpy as np
import cv2
import os
import logging
from typing import Tuple, Optional, Dict, List
import pickle
import glob
from PIL import Image

class SimpleFaceRecognitionSystem:
    """
    Prosty system rozpoznawania twarzy używający biblioteki face-recognition
    Zamiennik dla DeepFace gdy są problemy z TensorFlow
    """
    
    def __init__(self, tolerance: float = 0.6):
        """
        Inicjalizacja systemu
        
        Args:
            tolerance: Próg podobieństwa (0.0-1.0, mniejsze = bardziej restrykcyjne)
        """
        self.known_face_encodings = {}  # user_id -> list of encodings
        self.tolerance = tolerance
        self.user_photos_paths = {}  # user_id -> list of photo paths
        
        logging.info(f"SimpleFaceRecognitionSystem initialized with tolerance={tolerance}")
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Wstępne przetwarzanie obrazu"""
        try:
            if not os.path.exists(image_path):
                logging.error(f"File not found: {image_path}")
                return None
            
            # face-recognition preferuje RGB
            image = face_recognition.load_image_file(image_path)
            
            # Sprawdź rozmiar
            h, w = image.shape[:2]
            if h < 50 or w < 50:
                # Powiększ jeśli za mały
                pil_image = Image.fromarray(image)
                pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
                image = np.array(pil_image)
                logging.warning(f"Image {image_path} resized to 224x224")
            
            return image
            
        except Exception as e:
            logging.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def extract_face_encoding(self, image_path: str) -> Optional[np.ndarray]:
        """Ekstraktuje encoding twarzy z obrazu"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return None
            
            # Znajdź lokalizacje twarzy
            face_locations = face_recognition.face_locations(image, model="hog")  # "hog" szybszy niż "cnn"
            
            if len(face_locations) == 0:
                logging.warning(f"No face found in {image_path}")
                return None
            
            # Ekstraktuj encodingi
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) > 0:
                logging.debug(f"Successfully extracted face encoding from {image_path}")
                return face_encodings[0]  # Weź pierwszą twarz
            else:
                logging.warning(f"Could not encode face in {image_path}")
                return None
                
        except Exception as e:
            logging.error(f"Error extracting face encoding from {image_path}: {e}")
            return None
    
    def load_face_database(self, photos_directory: str) -> bool:
        """Wczytuje bazę danych twarzy z katalogu"""
        if not os.path.exists(photos_directory):
            logging.error(f"Photos directory does not exist: {photos_directory}")
            return False
        
        self.known_face_encodings = {}
        self.user_photos_paths = {}
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        user_folders = [f for f in os.listdir(photos_directory) 
                       if os.path.isdir(os.path.join(photos_directory, f))]
        
        if len(user_folders) < 10:
            logging.warning(f"Found only {len(user_folders)} users, minimum 10 required")
        
        total_encodings = 0
        
        for user_folder in user_folders:
            user_path = os.path.join(photos_directory, user_folder)
            user_id = user_folder
            
            # Znajdź wszystkie zdjęcia
            photo_paths = []
            for ext in supported_formats:
                pattern = os.path.join(user_path, f"*{ext}")
                photo_paths.extend(glob.glob(pattern))
                pattern = os.path.join(user_path, f"*{ext.upper()}")
                photo_paths.extend(glob.glob(pattern))
            
            if len(photo_paths) < 5:
                logging.warning(f"User {user_id} has only {len(photo_paths)} photos, minimum 5 required")
            
            user_encodings = []
            valid_photos = []
            
            for photo_path in photo_paths:
                encoding = self.extract_face_encoding(photo_path)
                if encoding is not None:
                    user_encodings.append(encoding)
                    valid_photos.append(photo_path)
                    total_encodings += 1
            
            if user_encodings:
                self.known_face_encodings[user_id] = user_encodings
                self.user_photos_paths[user_id] = valid_photos
                logging.info(f"Loaded {len(user_encodings)} face encodings for user {user_id}")
            else:
                logging.warning(f"No valid face encodings found for user {user_id}")
        
        logging.info(f"Face database loaded: {len(self.known_face_encodings)} users, {total_encodings} total encodings")
        return len(self.known_face_encodings) > 0
    
    def recognize_face(self, image_path: str) -> Tuple[Optional[str], float]:
        """Rozpoznaje tożsamość z obrazu"""
        if not self.known_face_encodings:
            logging.error("Face database is empty. Load database first.")
            return None, 0.0
        
        # Ekstraktuj encoding z obrazu testowego
        test_encoding = self.extract_face_encoding(image_path)
        if test_encoding is None:
            logging.warning(f"No face found in test image: {image_path}")
            return None, 0.0
        
        best_user = None
        best_distance = float('inf')
        all_distances = []
        
        # Porównaj z wszystkimi użytkownikami
        for user_id, user_encodings in self.known_face_encodings.items():
            # Oblicz odległości do wszystkich zdjęć użytkownika
            distances = face_recognition.face_distance(user_encodings, test_encoding)
            min_distance = np.min(distances)
            all_distances.append((user_id, min_distance))
            
            if min_distance < best_distance:
                best_distance = min_distance
                best_user = user_id
        
        # Oblicz poziom pewności
        confidence = self._calculate_confidence(best_distance, all_distances)
        
        # Sprawdź próg
        if best_distance <= self.tolerance:
            logging.info(f"Face recognized as {best_user} with confidence {confidence:.3f} (distance: {best_distance:.3f})")
            return best_user, confidence
        else:
            logging.info(f"Face not recognized. Best match: {best_user} with distance {best_distance:.3f} > tolerance {self.tolerance}")
            return None, confidence
    
    def _calculate_confidence(self, best_distance: float, all_distances: List[Tuple[str, float]]) -> float:
        """Oblicza poziom pewności na podstawie odległości"""
        if not all_distances:
            return 0.0
        
        # Sortuj odległości
        distances = [d[1] for d in all_distances]
        distances.sort()
        
        # Prosta konwersja odległości na pewność
        # face_recognition distance: 0.0 = identyczne, >0.6 = różne
        confidence = max(0.0, 1.0 - (best_distance / self.tolerance))
        
        # Jeśli mamy więcej użytkowników, uwzględnij separację
        if len(distances) > 1:
            second_best = distances[1]
            if second_best > 0:
                separation = (second_best - best_distance) / second_best
                confidence = confidence * (0.7 + 0.3 * separation)
        
        return min(1.0, max(0.0, confidence))
    
    def save_database(self, filepath: str) -> bool:
        """Zapisuje bazę danych twarzy do pliku"""
        try:
            data = {
                'known_face_encodings': {user_id: [enc.tolist() for enc in encodings] 
                                       for user_id, encodings in self.known_face_encodings.items()},
                'user_photos_paths': self.user_photos_paths,
                'tolerance': self.tolerance
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logging.info(f"Face database saved to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving face database: {e}")
            return False
    
    def load_database(self, filepath: str) -> bool:
        """Wczytuje bazę danych twarzy z pliku"""
        try:
            if not os.path.exists(filepath):
                logging.error(f"Database file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.known_face_encodings = {user_id: [np.array(enc) for enc in encodings] 
                                       for user_id, encodings in data['known_face_encodings'].items()}
            self.user_photos_paths = data['user_photos_paths']
            self.tolerance = data['tolerance']
            
            logging.info(f"Face database loaded from {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading face database: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Zwraca statystyki bazy danych"""
        total_encodings = sum(len(encodings) for encodings in self.known_face_encodings.values())
        
        return {
            'total_users': len(self.known_face_encodings),
            'total_encodings': total_encodings,
            'average_photos_per_user': total_encodings / len(self.known_face_encodings) if self.known_face_encodings else 0,
            'tolerance': self.tolerance,
            'library': 'face-recognition'
        }


# Test face-recognition
def test_simple_face_recognition():
    """Test systemu face-recognition"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Inicjalizacja
    face_system = SimpleFaceRecognitionSystem(tolerance=0.6)
    
    # Ścieżki
    photos_dir = "face_photos"
    database_file = "simple_face_database.pkl"
    
    if not os.path.exists(photos_dir):
        print(f"Tworzę strukturę katalogów w {photos_dir}")
        os.makedirs(photos_dir, exist_ok=True)
        for i in range(1, 11):
            user_dir = os.path.join(photos_dir, f"user{i:02d}")
            os.makedirs(user_dir, exist_ok=True)
        print("Dodaj zdjęcia użytkowników i uruchom ponownie.")
        return
    
    # Wczytaj bazę
    print("Wczytywanie bazy danych twarzy...")
    success = face_system.load_face_database(photos_dir)
    
    if success:
        stats = face_system.get_stats()
        print(f"Statystyki bazy danych:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Zapisz bazę
        face_system.save_database(database_file)
        print("✅ System face-recognition działa!")
        
    else:
        print("❌ Błąd wczytywania bazy danych")


if __name__ == "__main__":
    test_simple_face_recognition()