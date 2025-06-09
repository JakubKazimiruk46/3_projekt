# live_multimodal_webcam.py
"""
Live multimodal authentication z kamery internetowej
1. Wykrywa i rozpoznaje twarz użytkownika
2. Czeka na pokazanie podpisu na kartce
3. Przeprowadza autentykację multimodalną na żywo
"""

import cv2
import numpy as np
import time
import os
import logging
from PIL import Image
import tempfile
from multimodal_biometric_system import MultimodalBiometricSystem

class LiveMultimodalAuth:
    """System live authentication z kamery"""
    
    def __init__(self, multimodal_system):
        self.multimodal_system = multimodal_system
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Stany procesu
        self.state = "WAITING_FOR_FACE"  # WAITING_FOR_FACE, FACE_CAPTURED, WAITING_FOR_SIGNATURE, AUTHENTICATING
        self.face_image = None
        self.signature_image = None
        self.face_capture_time = None
        self.signature_capture_time = None
        
        # Parametry
        self.face_stability_time = 2.0  # Sekundy stabilnej twarzy przed przechwyceniem
        self.signature_capture_delay = 3.0  # Sekundy na pokazanie podpisu
        
        # Historia wykrytych twarzy (dla stabilności)
        self.face_history = []
        self.max_face_history = 10
        
        logging.info("Live Multimodal Authentication initialized")
    
    def detect_faces(self, frame):
        """Wykrywa twarze w ramce"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def is_face_stable(self, faces):
        """Sprawdza czy twarz jest stabilna przez określony czas"""
        if len(faces) != 1:
            self.face_history = []  # Reset jeśli nie ma dokładnie jednej twarzy
            return False
        
        # Dodaj aktualną twarz do historii
        face = faces[0]
        current_time = time.time()
        self.face_history.append((current_time, face))
        
        # Usuń stare wpisy
        cutoff_time = current_time - self.face_stability_time
        self.face_history = [(t, f) for t, f in self.face_history if t > cutoff_time]
        
        # Sprawdź czy mamy wystarczająco długą historię
        if len(self.face_history) < self.max_face_history:
            return False
        
        # Sprawdź stabilność pozycji
        positions = [f for _, f in self.face_history]
        avg_x = np.mean([x for x, y, w, h in positions])
        avg_y = np.mean([y for x, y, w, h in positions])
        
        # Sprawdź czy wszystkie pozycje są blisko średniej
        tolerance = 20  # piksele
        for x, y, w, h in positions:
            if abs(x - avg_x) > tolerance or abs(y - avg_y) > tolerance:
                return False
        
        return True
    
    def capture_face(self, frame, face_box):
        """Przechwytuje obraz twarzy"""
        x, y, w, h = face_box
        
        # Powiększ obszar o 20% dla lepszego kontekstu
        margin = int(0.2 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        face_crop = frame[y1:y2, x1:x2]
        
        # Zapisz do pliku tymczasowego
        temp_face_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(temp_face_path, face_crop)
        
        self.face_image = temp_face_path
        self.face_capture_time = time.time()
        
        logging.info(f"Face captured and saved to {temp_face_path}")
        return temp_face_path
    
    def detect_signature_area(self, frame):
        """Wykrywa obszar podpisu (uproszczona metoda)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Wykryj krawędzie (podpis na białej kartce)
        edges = cv2.Canny(gray, 50, 150)
        
        # Znajdź kontury
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Znajdź największy prostokątny kontur (kartka)
        largest_area = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area and area > 5000:  # Minimum area threshold
                # Aproksymuj do prostokąta
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Prostokąt
                    largest_area = area
                    best_contour = approx
        
        if best_contour is not None:
            # Zwróć bounding box
            x, y, w, h = cv2.boundingRect(best_contour)
            return (x, y, w, h)
        
        return None
    
    def capture_signature(self, frame):
        """Przechwytuje obraz podpisu"""
        signature_area = self.detect_signature_area(frame)
        
        if signature_area is None:
            # Fallback - użyj środkową część ekranu
            h, w = frame.shape[:2]
            x, y = w//4, h//4
            w, h = w//2, h//2
            signature_area = (x, y, w, h)
        
        x, y, w, h = signature_area
        signature_crop = frame[y:y+h, x:x+w]
        
        # Zapisz do pliku tymczasowego
        temp_sig_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(temp_sig_path, signature_crop)
        
        self.signature_image = temp_sig_path
        self.signature_capture_time = time.time()
        
        logging.info(f"Signature captured and saved to {temp_sig_path}")
        return temp_sig_path, signature_area
    
    def authenticate(self):
        """Przeprowadza autentykację multimodalną"""
        if not self.face_image or not self.signature_image:
            return None
        
        try:
            result = self.multimodal_system.recognize_multimodal(
                self.face_image, 
                self.signature_image
            )
            
            # Posprzątaj pliki tymczasowe
            if os.path.exists(self.face_image):
                os.remove(self.face_image)
            if os.path.exists(self.signature_image):
                os.remove(self.signature_image)
            
            return result
            
        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return None
    
    def reset_state(self):
        """Resetuje stan systemu"""
        self.state = "WAITING_FOR_FACE"
        self.face_image = None
        self.signature_image = None
        self.face_capture_time = None
        self.signature_capture_time = None
        self.face_history = []
    
    def draw_ui(self, frame):
        """Rysuje interfejs użytkownika na ramce"""
        h, w = frame.shape[:2]
        
        # Tło dla instrukcji
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Instrukcje w zależności od stanu
        if self.state == "WAITING_FOR_FACE":
            instruction = "Pokaz twarz do kamery i nie ruszaj sie przez 2 sekundy"
            color = (0, 255, 255)  # Żółty
            
            # Stabilność twarzy
            stability = len(self.face_history) / self.max_face_history
            cv2.rectangle(frame, (20, 90), (20 + int(300 * stability), 110), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 90), (320, 110), (255, 255, 255), 2)
            
        elif self.state == "FACE_CAPTURED":
            instruction = "Twarz przechwycona! Pokaz podpis na kartce"
            color = (0, 255, 0)  # Zielony
            
        elif self.state == "WAITING_FOR_SIGNATURE":
            instruction = "Pokaz podpis wyraznie do kamery"
            color = (255, 255, 0)  # Cyan
            
        elif self.state == "AUTHENTICATING":
            instruction = "Autentykacja w toku..."
            color = (255, 0, 255)  # Magenta
        
        # Wyświetl instrukcję
        cv2.putText(frame, instruction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Status
        status_text = f"Stan: {self.state}"
        cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Główna pętla live authentication"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logging.error("Cannot open webcam")
            return
        
        logging.info("Live Multimodal Authentication started")
        print("\n🔴 LIVE MULTIMODAL AUTHENTICATION")
        print("=" * 50)
        print("Instrukcje:")
        print("1. Pokaż twarz do kamery (nie ruszaj się 2 sekundy)")
        print("2. Po przechwyceniu twarzy pokaż podpis na kartce")
        print("3. System przeprowadzi autentykację")
        print("Naciśnij 'q' aby wyjść, 'r' aby zresetować")
        print("=" * 50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Odbicie lustrzane dla lepszego UX
                frame = cv2.flip(frame, 1)
                
                # Wykryj twarze
                faces = self.detect_faces(frame)
                
                # Rysuj ramki wokół twarzy
                for (x, y, w, h) in faces:
                    color = (0, 255, 0) if len(faces) == 1 else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # State machine
                if self.state == "WAITING_FOR_FACE":
                    if self.is_face_stable(faces):
                        # Przechwytuj twarz
                        self.capture_face(frame, faces[0])
                        self.state = "FACE_CAPTURED"
                        logging.info("Face captured, waiting for signature")
                
                elif self.state == "FACE_CAPTURED":
                    # Automatyczne przejście do oczekiwania na podpis po krótkiej przerwie
                    if time.time() - self.face_capture_time > 1.0:
                        self.state = "WAITING_FOR_SIGNATURE"
                
                elif self.state == "WAITING_FOR_SIGNATURE":
                    # Czekaj na naciśnięcie spacji lub automatyczne wykrycie
                    signature_area = self.detect_signature_area(frame)
                    if signature_area:
                        x, y, w, h = signature_area
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                        cv2.putText(frame, "Nacisnij SPACJA aby przechwyc podpis", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                elif self.state == "AUTHENTICATING":
                    # Przeprowadź autentykację
                    result = self.authenticate()
                    
                    if result:
                        # Wyświetl wyniki
                        self.display_result(frame, result)
                        cv2.imshow('Live Multimodal Authentication', frame)
                        cv2.waitKey(3000)  # Pokaż wynik przez 3 sekundy
                    
                    # Reset po autentykacji
                    self.reset_state()
                
                # Rysuj UI
                frame = self.draw_ui(frame)
                
                cv2.imshow('Live Multimodal Authentication', frame)
                
                # Obsługa klawiszy
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_state()
                    logging.info("State reset")
                elif key == ord(' ') and self.state == "WAITING_FOR_SIGNATURE":
                    # Przechwytuj podpis
                    sig_path, sig_area = self.capture_signature(frame)
                    self.state = "AUTHENTICATING"
                    logging.info("Signature captured, starting authentication")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Live authentication stopped")
    
    def display_result(self, frame, result):
        """Wyświetla wyniki autentykacji na ramce"""
        h, w = frame.shape[:2]
        
        # Tło dla wyników
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (w-50, h-50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Tytuł
        cv2.putText(frame, "WYNIK AUTENTYKACJI", (w//2-150, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Wyniki
        y_pos = 150
        line_height = 40
        
        lines = [
            f"Twarz: {result['face_user']} ({result['face_confidence']:.3f})",
            f"Podpis: {result['signature_user']} ({result['signature_confidence']:.3f})",
            f"Laczna pewnosc: {result['combined_confidence']:.3f}",
            f"Decyzja: {result['final_decision'] or 'BRAK ROZPOZNANIA'}",
            f"Powod: {result['decision_reason']}"
        ]
        
        for line in lines:
            cv2.putText(frame, line, (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += line_height
        
        # Status - zielony jeśli sukces, czerwony jeśli odmowa
        decision_color = (0, 255, 0) if result['final_decision'] else (0, 0, 255)
        status_text = "DOSTEP PRZYZNANY" if result['final_decision'] else "DOSTEP ODMOWIONY"
        
        cv2.putText(frame, status_text, (w//2-100, y_pos + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, decision_color, 3)


def run_live_authentication():
    """Uruchamia live authentication"""
    
    # Sprawdź wymagania
    required_files = ["mlp_model_multi.pkl", "scaler_multi.pkl"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Brak wymaganych modeli:")
        for f in missing_files:
            print(f"  - {f}")
        print("Uruchom najpierw trenowanie modeli podpisów!")
        return False
    
    if not os.path.exists("face_photos"):
        print("❌ Brak katalogu face_photos")
        print("Dodaj zdjęcia użytkowników do face_photos/")
        return False
    
    try:
        # Inicjalizacja systemu multimodalnego
        print("Inicjalizacja systemu multimodalnego...")
        multimodal_system = MultimodalBiometricSystem(
            w_face=0.6,
            w_signature=0.4,
            confidence_threshold=0.5
        )
        
        # Konfiguracja
        face_success = multimodal_system.setup_face_system("face_photos", "face_database.json")
        signature_success = multimodal_system.setup_signature_system()
        
        if not (face_success and signature_success):
            print("❌ Błąd konfiguracji systemu")
            return False
        
        print("✅ System multimodalny skonfigurowany!")
        
        # Uruchom live authentication
        live_auth = LiveMultimodalAuth(multimodal_system)
        live_auth.run()
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd live authentication: {e}")
        logging.exception("Live authentication error")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_live_authentication()