import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SpoofingDetectionSystem:
    def __init__(self):
        self.multimodal_system = None
        self.spoofing_detector = None
        self.scaler_spoofing = StandardScaler()
        
    def create_spoofing_dataset(self, legitimate_path, spoofing_path):
        """Stwórz zbiór danych z prawdziwymi i sfałszowanymi próbkami"""
        
        features_list = []
        labels_list = []
        spoofing_labels = []  # 0=genuine, 1=spoofed
        
        # Prawdziwe próbki (genuine)
        print("Ładowanie prawdziwych próbek...")
        genuine_features, genuine_labels = self._load_legitimate_data(legitimate_path)
        
        if len(genuine_features) > 0:
            features_list.extend(genuine_features)
            labels_list.extend(genuine_labels)
            spoofing_labels.extend([0] * len(genuine_features))  # 0 = genuine
        
        # Sfałszowane próbki (spoofed)
        print("Ładowanie sfałszowanych próbek...")
        spoofed_features, spoofed_labels = self._load_spoofing_data(spoofing_path)
        
        if len(spoofed_features) > 0:
            features_list.extend(spoofed_features)
            labels_list.extend(spoofed_labels)
            spoofing_labels.extend([1] * len(spoofed_features))  # 1 = spoofed
        
        return np.array(features_list), np.array(labels_list), np.array(spoofing_labels)
    
    def _load_legitimate_data(self, legitimate_path):
        """Wczytaj prawdziwe pary twarza-podpis"""
        from multimodal_biometric import MultimodalBiometricSystem
        
        face_path = os.path.join(legitimate_path, "faces")
        signature_path = os.path.join(legitimate_path, "signatures")
        
        if not (os.path.exists(face_path) and os.path.exists(signature_path)):
            print(f"Brak katalogów: {face_path} lub {signature_path}")
            return [], []
        
        multimodal = MultimodalBiometricSystem()
        features_dict, labels = multimodal.load_multimodal_dataset(face_path, signature_path)
        
        # Użyj cech kombinowanych
        if 'combined' in features_dict and len(features_dict['combined']) > 0:
            return features_dict['combined'].tolist(), labels.tolist()
        
        return [], []
    
    def _load_spoofing_data(self, spoofing_path):
        """Wczytaj sfałszowane próbki (np. dobra twarz + zły podpis)"""
        from multimodal_biometric import MultimodalBiometricSystem
        
        # Różne typy ataków
        attack_types = [
            "face_correct_signature_wrong",  # dobra twarz + zły podpis
            "face_wrong_signature_correct",  # zła twarz + dobry podpis
            "both_wrong"                     # zła twarz + zły podpis
        ]
        
        all_features = []
        all_labels = []
        
        for attack_type in attack_types:
            attack_path = os.path.join(spoofing_path, attack_type)
            face_path = os.path.join(attack_path, "faces")
            signature_path = os.path.join(attack_path, "signatures")
            
            if os.path.exists(face_path) and os.path.exists(signature_path):
                print(f"  Ładowanie ataków typu: {attack_type}")
                
                multimodal = MultimodalBiometricSystem()
                features_dict, labels = multimodal.load_multimodal_dataset(face_path, signature_path)
                
                if 'combined' in features_dict and len(features_dict['combined']) > 0:
                    all_features.extend(features_dict['combined'].tolist())
                    # Etykieta pokazuje kogo atakujący próbuje udawać
                    all_labels.extend(labels.tolist())
        
        return all_features, all_labels
    
    def train_spoofing_detector(self, features, spoofing_labels):
        """Trenuj detektor spoofingu (binarny klasyfikator)"""
        print("\nTrenowanie detektora spoofingu...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, spoofing_labels, test_size=0.2, random_state=42
        )
        
        # Normalizacja
        X_train_scaled = self.scaler_spoofing.fit_transform(X_train)
        X_test_scaled = self.scaler_spoofing.transform(X_test)
        
        # Model binarny (genuine vs spoofed)
        self.spoofing_detector = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=2000,
            random_state=42,
            early_stopping=False
        )
        
        self.spoofing_detector.fit(X_train_scaled, y_train)
        y_pred = self.spoofing_detector.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"Dokładność detektora spoofingu: {accuracy:.2f}%")
        
        # Raport szczegółowy
        print("\nRaport detektora spoofingu:")
        unique_classes = np.unique(y_test)
        target_names = ['Genuine', 'Spoofed']
        
        if len(unique_classes) == 2:
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        else:
            print(classification_report(y_test, y_pred, zero_division=0))
            print(f"UWAGA: Tylko {len(unique_classes)} klasa w zbiorze testowym!")
        
        # Macierz pomyłek - tylko jeśli mamy dane
        if len(spoofing_labels) > 1 and len(np.unique(spoofing_labels)) > 1:
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                       xticklabels=target_names, yticklabels=target_names)
            plt.title("Macierz pomyłek - Detektor spoofingu")
            plt.xlabel("Przewidziane")
            plt.ylabel("Rzeczywiste")
            plt.savefig("spoofing_detection_confusion_matrix.png")
            plt.close()
        else:
            print("Pomiń macierz pomyłek - za mało danych spoofingowych")
        
        return accuracy
    
    def evaluate_system_security(self, features, identity_labels, spoofing_labels):
        """Ocena bezpieczeństwa systemu wobec ataków"""
        print("\n" + "="*50)
        print("ANALIZA BEZPIECZEŃSTWA SYSTEMU")
        print("="*50)
        
        # 1. Skuteczność rozpoznawania na genuine samples
        genuine_mask = spoofing_labels == 0
        genuine_features = features[genuine_mask]
        genuine_labels = identity_labels[genuine_mask]
        
        if len(genuine_features) > 0:
            print(f"\n1. Próbki prawdziwe: {len(genuine_features)}")
            # Tu można dodać test na genuine samples
        
        # 2. Podatność na ataki spoofingu
        spoofed_mask = spoofing_labels == 1
        spoofed_features = features[spoofed_mask]
        spoofed_labels_identity = identity_labels[spoofed_mask]
        
        if len(spoofed_features) > 0:
            print(f"2. Próbki sfałszowane: {len(spoofed_features)}")
            
            # Test: czy system błędnie akceptuje sfałszowane próbki?
            spoofing_proba = self.spoofing_detector.predict_proba(
                self.scaler_spoofing.transform(spoofed_features)
            )
            
            # Prawdopodobieństwo zaklasyfikowania jako genuine
            false_acceptance_rate = np.mean(spoofing_proba[:, 0] > 0.5) * 100
            print(f"   False Acceptance Rate: {false_acceptance_rate:.2f}%")
        
        # 3. Analiza typów ataków
        self._analyze_attack_types(features, identity_labels, spoofing_labels)
    
    def _analyze_attack_types(self, features, identity_labels, spoofing_labels):
        """Analiza skuteczności różnych typów ataków"""
        print("\n3. ANALIZA TYPÓW ATAKÓW:")
        print("-" * 30)
        
        spoofed_features = features[spoofing_labels == 1]
        
        if len(spoofed_features) > 0:
            # Przewidywania detektora spoofingu
            spoofing_pred = self.spoofing_detector.predict(
                self.scaler_spoofing.transform(spoofed_features)
            )
            
            detected_attacks = np.sum(spoofing_pred == 1)
            missed_attacks = np.sum(spoofing_pred == 0)
            
            print(f"Wykryte ataki: {detected_attacks}/{len(spoofed_features)}")
            print(f"Pominięte ataki: {missed_attacks}/{len(spoofed_features)}")
            print(f"Detection Rate: {(detected_attacks/len(spoofed_features)*100):.2f}%")

# Funkcja główna do testowania bezpieczeństwa
def run_security_evaluation():
    """Uruchom pełną ewaluację bezpieczeństwa"""
    
    spoofing_system = SpoofingDetectionSystem()
    
    # Ścieżki do danych
    legitimate_path = "legitimate_data"  # prawdziwe pary
    spoofing_path = "spoofing_attacks"   # sfałszowane próbki
    
    # Sprawdź strukturę katalogów
    required_paths = [
        os.path.join(legitimate_path, "faces"),
        os.path.join(legitimate_path, "signatures"),
        os.path.join(spoofing_path, "face_correct_signature_wrong", "faces"),
        os.path.join(spoofing_path, "face_correct_signature_wrong", "signatures")
    ]
    
    missing_paths = [p for p in required_paths if not os.path.exists(p)]
    if missing_paths:
        print("Brak wymaganych katalogów:")
        for path in missing_paths:
            print(f"   {path}")
        print("\nStruktura powinna być:")
        print("legitimate_data/")
        print("├── faces/user01/face1.jpg")
        print("└── signatures/user01/sig1.jpg")
        print("spoofing_attacks/")
        print("├── face_correct_signature_wrong/")
        print("│   ├── faces/user01/face1.jpg")  # dobra twarz
        print("│   └── signatures/user02/sig1.jpg")  # zły podpis
        print("└── ...")
        return
    
    # Wczytaj dane
    features, identity_labels, spoofing_labels = spoofing_system.create_spoofing_dataset(
        legitimate_path, spoofing_path
    )
    
    if len(features) == 0:
        print("Nie udało się wczytać danych!")
        return
    
    print(f"Załadowano {len(features)} próbek")
    print(f"   Genuine: {np.sum(spoofing_labels == 0)}")
    print(f"   Spoofed: {np.sum(spoofing_labels == 1)}")
    
    # Trenuj detektor spoofingu
    spoofing_accuracy = spoofing_system.train_spoofing_detector(features, spoofing_labels)
    
    # Ewaluacja bezpieczeństwa
    spoofing_system.evaluate_system_security(features, identity_labels, spoofing_labels)
    
    print(f"\nOCENA BEZPIECZEŃSTWA ZAKOŃCZONA")
    print(f"Dokładność detektora spoofingu: {spoofing_accuracy:.2f}%")

if __name__ == "__main__":
    run_security_evaluation()