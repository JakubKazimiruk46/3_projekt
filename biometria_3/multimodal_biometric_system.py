# multimodal_biometric_system.py
import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import json
from PIL import Image
import time

# Import własnych modułów
from face_recognition_system import SimpleFaceRecognitionSystem
from signature_recognition import preprocess_image, extract_features


class SignatureRecognitionAdapter:
    """
    Adapter dla istniejącego systemu rozpoznawania podpisów
    Umożliwia łatwe użycie w systemie multimodalnym
    """
    
    def __init__(self, model_path: str = "mlp_model_multi.pkl", 
                 scaler_path: str = "scaler_multi.pkl"):
        """
        Inicjalizacja adaptera systemu podpisów
        
        Args:
            model_path: Ścieżka do zapisanego modelu MLP
            scaler_path: Ścieżka do zapisanego scalera
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Wczytuje zapisany model i scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_loaded = True
                logging.info(f"Signature model loaded from {self.model_path}")
                return True
            else:
                logging.error(f"Signature model files not found: {self.model_path}, {self.scaler_path}")
                return False
        except Exception as e:
            logging.error(f"Error loading signature model: {e}")
            return False
    
    def recognize_signature(self, signature_path: str) -> Tuple[Optional[str], float]:
        """
        Rozpoznaje podpis i zwraca tożsamość + poziom pewności
        
        Args:
            signature_path: Ścieżka do obrazu podpisu
            
        Returns:
            Tuple (user_id, confidence)
        """
        if not self.is_loaded:
            if not self.load_model():
                return None, 0.0
        
        try:
            # Wstępne przetwarzanie podpisu
            processed_signature = preprocess_image(signature_path)
            if processed_signature is None:
                logging.warning(f"Could not process signature: {signature_path}")
                return None, 0.0
            
            # Ekstrakcja cech
            features = extract_features(processed_signature)
            features_array = np.array(features).reshape(1, -1)
            
            # Normalizacja cech
            features_scaled = self.scaler.transform(features_array)
            
            # Predykcja
            prediction = self.model.predict(features_scaled)[0]
            
            # Oblicz poziom pewności (prawdopodobieństwo klasy)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
            else:
                # Dla modeli bez predict_proba - użyj decision_function lub ustaw stałą wartość
                confidence = 0.8  # Domyślna wartość
            
            logging.info(f"Signature recognized as {prediction} with confidence {confidence:.3f}")
            return prediction, confidence
            
        except Exception as e:
            logging.error(f"Error recognizing signature {signature_path}: {e}")
            return None, 0.0


class MultimodalBiometricSystem:
    """
    System multimodalny łączący rozpoznawanie twarzy i podpisu
    Implementuje logikę zadania z wagami i progami decyzyjnymi
    """
    
    def __init__(self, 
                 w_face: float = 0.6, 
                 w_signature: float = 0.4,
                 confidence_threshold: float = 0.5):
        """
        Inicjalizacja systemu multimodalnego
        
        Args:
            w_face: Waga dla rozpoznawania twarzy
            w_signature: Waga dla rozpoznawania podpisu  
            confidence_threshold: Próg pewności ogólnej (domyślnie 0.5)
        """
        # Sprawdź czy wagi sumują się do 1
        if abs(w_face + w_signature - 1.0) > 1e-6:
            raise ValueError(f"Wagi muszą sumować się do 1.0, otrzymano: {w_face} + {w_signature} = {w_face + w_signature}")
        
        self.w_face = w_face
        self.w_signature = w_signature
        self.confidence_threshold = confidence_threshold
        
        # Inicjalizacja systemów
        self.face_system = SimpleFaceRecognitionSystem()
        self.signature_system = SignatureRecognitionAdapter()
        
        logging.info(f"Multimodal system initialized: w_face={w_face}, w_signature={w_signature}, threshold={confidence_threshold}")
    
    def setup_face_system(self, photos_directory: str, database_file: str = None) -> bool:
        """
        Konfiguruje system rozpoznawania twarzy
        
        Args:
            photos_directory: Katalog ze zdjęciami użytkowników
            database_file: Opcjonalny plik z zapisaną bazą danych
            
        Returns:
            True jeśli sukces
        """
        if database_file and os.path.exists(database_file):
            success = self.face_system.load_database(database_file)
        else:
            success = self.face_system.load_face_database(photos_directory)
            if success and database_file:
                self.face_system.save_database(database_file)
        
        if success:
            stats = self.face_system.get_stats()
            logging.info(f"Face system ready: {stats['total_users']} users, {stats['total_encodings']} encodings")
        
        return success
    
    def setup_signature_system(self, model_path: str = "mlp_model_multi.pkl", 
                              scaler_path: str = "scaler_multi.pkl") -> bool:
        """
        Konfiguruje system rozpoznawania podpisów
        
        Args:
            model_path: Ścieżka do modelu podpisów
            scaler_path: Ścieżka do scalera
            
        Returns:
            True jeśli sukces
        """
        self.signature_system = SignatureRecognitionAdapter(model_path, scaler_path)
        success = self.signature_system.load_model()
        
        if success:
            logging.info("Signature system ready")
        
        return success
    
    def recognize_multimodal(self, face_image_path: str, signature_image_path: str) -> Dict:
        """
        Główna funkcja rozpoznawania multimodalnego
        
        Args:
            face_image_path: Ścieżka do zdjęcia twarzy
            signature_image_path: Ścieżka do obrazu podpisu
            
        Returns:
            Dict z wynikami rozpoznawania
        """
        result = {
            'face_user': None,
            'face_confidence': 0.0,
            'signature_user': None, 
            'signature_confidence': 0.0,
            'combined_confidence': 0.0,
            'final_decision': None,
            'decision_reason': '',
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Rozpoznawanie twarzy
            logging.info("Rozpoznawanie twarzy...")
            face_user, face_conf = self.face_system.recognize_face(face_image_path)
            result['face_user'] = face_user
            result['face_confidence'] = face_conf
            
            # Rozpoznawanie podpisu
            logging.info("Rozpoznawanie podpisu...")
            sig_user, sig_conf = self.signature_system.recognize_signature(signature_image_path)
            result['signature_user'] = sig_user
            result['signature_confidence'] = sig_conf
            
            # Oblicz pewność łączną według wzoru z zadania
            # pewność_całość = w_face * pewność_face + w_signature * pewność_signature
            combined_conf = self.w_face * face_conf + self.w_signature * sig_conf
            result['combined_confidence'] = combined_conf
            
            # Logika decyzyjna według zadania
            if combined_conf > self.confidence_threshold:
                # Sprawdź czy oba systemy zwróciły tę samą tożsamość
                if face_user is not None and sig_user is not None and face_user == sig_user:
                    result['final_decision'] = face_user
                    result['decision_reason'] = f"Both systems agree: {face_user}"
                    logging.info(f"SUCCESS: Both systems recognized {face_user} (confidence: {combined_conf:.3f})")
                else:
                    result['final_decision'] = None
                    if face_user != sig_user:
                        result['decision_reason'] = f"Systems disagree: face={face_user}, signature={sig_user}"
                        logging.warning(f"DISAGREEMENT: Face={face_user}, Signature={sig_user}")
                    else:
                        result['decision_reason'] = "One or both systems failed to recognize"
                        logging.warning("FAILURE: One or both systems failed to recognize")
            else:
                result['final_decision'] = None
                result['decision_reason'] = f"Combined confidence too low: {combined_conf:.3f} <= {self.confidence_threshold}"
                logging.info(f"LOW CONFIDENCE: {combined_conf:.3f} <= {self.confidence_threshold}")
            
            result['processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logging.error(f"Error in multimodal recognition: {e}")
            result['decision_reason'] = f"Error: {str(e)}"
            result['processing_time'] = time.time() - start_time
            return result
    
    def evaluate_system(self, test_data: List[Tuple[str, str, str]]) -> Dict:
        """
        Ewaluacja skuteczności systemu multimodalnego
        
        Args:
            test_data: Lista tupli (face_path, signature_path, true_user_id)
            
        Returns:
            Dict ze statystykami skuteczności
        """
        results = {
            'total_tests': len(test_data),
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'no_predictions': 0,
            'face_only_correct': 0,
            'signature_only_correct': 0,
            'both_correct': 0,
            'face_only_accuracy': 0.0,
            'signature_only_accuracy': 0.0,
            'multimodal_accuracy': 0.0,
            'average_processing_time': 0.0,
            'detailed_results': []
        }
        
        total_time = 0.0
        face_correct = 0
        signature_correct = 0
        
        logging.info(f"Starting evaluation with {len(test_data)} test samples...")
        
        for i, (face_path, signature_path, true_user) in enumerate(test_data):
            logging.info(f"Testing sample {i+1}/{len(test_data)}: {true_user}")
            
            # Rozpoznawanie multimodalne
            result = self.recognize_multimodal(face_path, signature_path)
            total_time += result['processing_time']
            
            # Sprawdź poprawność poszczególnych systemów
            if result['face_user'] == true_user:
                face_correct += 1
                
            if result['signature_user'] == true_user:
                signature_correct += 1
                
            if result['face_user'] == true_user and result['signature_user'] == true_user:
                results['both_correct'] += 1
            
            # Sprawdź poprawność systemu multimodalnego
            if result['final_decision'] == true_user:
                results['correct_predictions'] += 1
            elif result['final_decision'] is None:
                results['no_predictions'] += 1
            else:
                results['incorrect_predictions'] += 1
            
            # Zapisz szczegółowe wyniki
            detailed_result = {
                'sample_id': i + 1,
                'true_user': true_user,
                'face_path': face_path,
                'signature_path': signature_path,
                **result
            }
            results['detailed_results'].append(detailed_result)
        
        # Oblicz statystyki
        results['face_only_correct'] = face_correct
        results['signature_only_correct'] = signature_correct
        results['face_only_accuracy'] = face_correct / len(test_data) * 100
        results['signature_only_accuracy'] = signature_correct / len(test_data) * 100
        results['multimodal_accuracy'] = results['correct_predictions'] / len(test_data) * 100
        results['average_processing_time'] = total_time / len(test_data)
        
        return results
    
    def print_evaluation_report(self, evaluation_results: Dict):
        """Wyświetla raport z ewaluacji systemu"""
        print("\n" + "="*60)
        print("RAPORT EWALUACJI SYSTEMU MULTIMODALNEGO")
        print("="*60)
        
        # Podstawowe statystyki
        print(f"\nPodstawowe statystyki:")
        print(f"  Łączna liczba testów: {evaluation_results['total_tests']}")
        print(f"  Poprawne rozpoznania: {evaluation_results['correct_predictions']}")
        print(f"  Niepoprawne rozpoznania: {evaluation_results['incorrect_predictions']}")
        print(f"  Brak rozpoznania: {evaluation_results['no_predictions']}")
        
        # Dokładność systemów
        print(f"\nDokładność systemów:")
        print(f"  Tylko twarz: {evaluation_results['face_only_accuracy']:.1f}%")
        print(f"  Tylko podpis: {evaluation_results['signature_only_accuracy']:.1f}%")
        print(f"  System multimodalny: {evaluation_results['multimodal_accuracy']:.1f}%")
        
        # Porównanie
        improvement_vs_face = evaluation_results['multimodal_accuracy'] - evaluation_results['face_only_accuracy']
        improvement_vs_signature = evaluation_results['multimodal_accuracy'] - evaluation_results['signature_only_accuracy']
        
        print(f"\nPoprawa względem systemów pojedynczych:")
        print(f"  vs. tylko twarz: {improvement_vs_face:+.1f}%")
        print(f"  vs. tylko podpis: {improvement_vs_signature:+.1f}%")
        
        # Wydajność
        print(f"\nWydajność:")
        print(f"  Średni czas przetwarzania: {evaluation_results['average_processing_time']:.3f}s")
        
        # Parametry systemu
        print(f"\nParametry systemu:")
        print(f"  Waga twarzy: {self.w_face}")
        print(f"  Waga podpisu: {self.w_signature}")
        print(f"  Próg pewności: {self.confidence_threshold}")
        
        print("="*60)
    
    def save_evaluation_results(self, evaluation_results: Dict, filepath: str):
        """Zapisuje wyniki ewaluacji do pliku JSON"""
        try:
            # Dodaj parametry systemu do wyników
            evaluation_results['system_parameters'] = {
                'w_face': self.w_face,
                'w_signature': self.w_signature,
                'confidence_threshold': self.confidence_threshold
            }
            
            with open(filepath, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logging.info(f"Evaluation results saved to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving evaluation results: {e}")
            return False


def create_test_dataset_from_test_data(test_data_dir: str, 
                                      user: str, photos_per_user: int = 2) -> List[Tuple[str, str, str]]:
    """
    Tworzy zestaw testowy z katalogu test_data
    
    Args:
        test_data_dir: Katalog test_data
        user: ID użytkownika (np. "user01")
        photos_per_user: Liczba testów na użytkownika
        
    Returns:
        Lista tupli (face_path, signature_path, user_id)
    """
    test_data = []
    
    if not os.path.exists(test_data_dir):
        logging.warning(f"Test data directory does not exist: {test_data_dir}")
        return test_data
    
    # Znajdź pliki testowe dla użytkownika
    face_patterns = [
        f"{user}_face_test.jpg",
        f"{user}_face_test.png", 
        f"{user}_face.jpg",
        f"{user}_face.png",
        "test_face.jpg",
        "test_face.png"
    ]
    
    signature_patterns = [
        f"{user}_signature_test.jpg",
        f"{user}_signature_test.png",
        f"{user}_signature.jpg", 
        f"{user}_signature.png",
        "test_signature.jpg",
        "test_signature.png"
    ]
    
    # Znajdź pliki twarzy
    face_files = []
    for pattern in face_patterns:
        face_path = os.path.join(test_data_dir, pattern)
        if os.path.exists(face_path):
            face_files.append(face_path)
    
    # Znajdź pliki podpisów
    signature_files = []
    for pattern in signature_patterns:
        sig_path = os.path.join(test_data_dir, pattern)
        if os.path.exists(sig_path):
            signature_files.append(sig_path)
    
    if not face_files:
        logging.warning(f"No face test files found for {user} in {test_data_dir}")
        logging.info(f"Expected files: {face_patterns}")
        return test_data
    
    if not signature_files:
        logging.warning(f"No signature test files found for {user} in {test_data_dir}")
        logging.info(f"Expected files: {signature_patterns}")
        return test_data
    
    # Utwórz pary testowe
    for i in range(min(photos_per_user, len(face_files), len(signature_files))):
        face_path = face_files[i]
        signature_path = signature_files[i]
        test_data.append((face_path, signature_path, user))
        logging.info(f"Created test pair: {os.path.basename(face_path)} + {os.path.basename(signature_path)}")
    
    logging.info(f"Created test dataset with {len(test_data)} samples for user {user} from test_data")
    return test_data


def main():
    """Funkcja główna - demonstracja systemu multimodalnego"""
    
    # Konfiguracja logowania
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("multimodal_system.log"),
            logging.StreamHandler()
        ]
    )
    
    print("SYSTEM BIOMETRYCZNY MULTIMODALNY")
    print("Łączy rozpoznawanie twarzy i podpisu")
    print("-" * 40)
    
    # Konfiguracja ścieżek (dostosuj do swojej struktury)
    face_photos_dir = "face_photos"  # Katalog ze zdjęciami twarzy
    signature_photos_dir = "sign_data/train"  # Katalog z podpisami (z signature_recognition.py)
    face_database_file = "face_database.json"
    
    # Parametry systemu (możesz eksperymentować)
    w_face = 0.6  # Waga dla twarzy
    w_signature = 0.4  # Waga dla podpisu
    confidence_threshold = 0.5  # Próg pewności
    
    # Inicjalizacja systemu
    multimodal_system = MultimodalBiometricSystem(
        w_face=w_face,
        w_signature=w_signature, 
        confidence_threshold=confidence_threshold
    )
    
    # Konfiguracja systemu twarzy
    print("Konfiguracja systemu rozpoznawania twarzy...")
    face_success = multimodal_system.setup_face_system(
        face_photos_dir, 
        face_database_file
    )
    
    if not face_success:
        print(f"BŁĄD: Nie można skonfigurować systemu twarzy.")
        print(f"Sprawdź czy istnieje katalog: {face_photos_dir}")
        print("Struktura powinna być:")
        print(f"{face_photos_dir}/")
        print("├── user01/")
        print("│   ├── photo1.jpg")
        print("│   └── ...")
        print("└── ...")
        return
    
    # Konfiguracja systemu podpisów
    print("Konfiguracja systemu rozpoznawania podpisów...")
    signature_success = multimodal_system.setup_signature_system()
    
    if not signature_success:
        print("BŁĄD: Nie można skonfigurować systemu podpisów.")
        print("Uruchom najpierw signature_recognition.py aby wytrenować model!")
        return
    
    print("✓ Systemy skonfigurowane pomyślnie!")
    
    # Test pojedynczy (jeśli masz przykładowe pliki)
    test_face = "test_face.jpg"
    test_signature = "test_signature.jpg"
    
    if os.path.exists(test_face) and os.path.exists(test_signature):
        print(f"\nTest pojedynczy:")
        print(f"  Twarz: {test_face}")
        print(f"  Podpis: {test_signature}")
        
        result = multimodal_system.recognize_multimodal(test_face, test_signature)
        
        print(f"\nWyniki:")
        print(f"  Twarz: {result['face_user']} (pewność: {result['face_confidence']:.3f})")
        print(f"  Podpis: {result['signature_user']} (pewność: {result['signature_confidence']:.3f})")
        print(f"  Łączna pewność: {result['combined_confidence']:.3f}")
        print(f"  DECYZJA: {result['final_decision'] or 'BRAK ROZPOZNANIA'}")
        print(f"  Powód: {result['decision_reason']}")
        print(f"  Czas: {result['processing_time']:.3f}s")
    
    # Ewaluacja systemu z plików test_data
    print(f"\nCzy uruchomić ewaluację systemu? (y/n): ", end="")
    if input().lower() in ['y', 'yes', 'tak']:
        
        # Test z plików w test_data/
        user = "user01"
        test_data_dir = "test_data"
        
        # Utwórz zestaw testowy z test_data
        test_data = create_test_dataset_from_test_data(
            test_data_dir, 
            user,
            photos_per_user=5  # Sprawdź do 5 plików testowych
        )
        
        if test_data:
            print(f"Uruchamianie ewaluacji z {len(test_data)} próbkami...")
            
            # Ewaluacja
            evaluation_results = multimodal_system.evaluate_system(test_data)
            
            # Raport
            multimodal_system.print_evaluation_report(evaluation_results)
            
            # Zapis wyników
            multimodal_system.save_evaluation_results(
                evaluation_results, 
                "multimodal_evaluation_results.json"
            )
            
            print("\nWyniki zapisane do: multimodal_evaluation_results.json")
        else:
            print("Brak danych testowych do ewaluacji.")
    
    print("\nZakończono działanie systemu multimodalnego.")


if __name__ == "__main__":
    main()