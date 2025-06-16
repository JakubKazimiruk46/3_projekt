import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from face_recognition_system import FaceRecognitionSystem
from signature_recognition import preprocess_image, extract_features


class MultimodalBiometricSystem:
    def __init__(self):
        self.face_system = FaceRecognitionSystem()
        self.scaler_combined = StandardScaler()
        self.scaler_face = StandardScaler()
        self.scaler_signature = StandardScaler()

        # Modele dla różnych fuzji
        self.combined_knn = None
        self.combined_mlp = None
        self.face_knn = None
        self.face_mlp = None
        self.signature_knn = None
        self.signature_mlp = None

    def extract_multimodal_features(self, face_path, signature_path):
        face_features = self.face_system.extract_face_features(face_path)

        signature_img = preprocess_image(signature_path)
        signature_features = extract_features(signature_img)

        if face_features is None:
            print(f"Nie udało się wyekstraktować cech twarzy z: {face_path}")
            return None

        combined_features = np.concatenate([face_features, signature_features])

        return {
            'combined': combined_features,
            'face': face_features,
            'signature': signature_features
        }

    def load_multimodal_dataset(self, face_dataset_path, signature_dataset_path):
        print("Ładowanie danych multimodalnych...")

        all_features = {'combined': [], 'face': [], 'signature': []}
        labels = []

        face_users = set(os.listdir(face_dataset_path))
        sig_users = set(os.listdir(signature_dataset_path))
        common_users = face_users.intersection(sig_users)

        print(f"Wspólni użytkownicy: {len(common_users)}")

        for user in common_users:
            face_user_path = os.path.join(face_dataset_path, user)
            sig_user_path = os.path.join(signature_dataset_path, user)

            if not (os.path.isdir(face_user_path) and os.path.isdir(sig_user_path)):
                continue

            face_files = [f for f in os.listdir(face_user_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            sig_files = [f for f in os.listdir(sig_user_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            min_files = min(len(face_files), len(sig_files))

            for i in range(min_files):
                face_path = os.path.join(face_user_path, face_files[i])
                sig_path = os.path.join(sig_user_path, sig_files[i])

                features = self.extract_multimodal_features(face_path, sig_path)

                if features is not None:
                    all_features['combined'].append(features['combined'])
                    all_features['face'].append(features['face'])
                    all_features['signature'].append(features['signature'])
                    labels.append(user)

        for key in all_features:
            all_features[key] = np.array(all_features[key])

        labels = np.array(labels)

        print(f"Załadowano {len(labels)} par twarza-podpis")
        return all_features, labels

    def train_all_models(self, features_dict, labels):
        """Trenuj wszystkie modele (pojedyncze modalności + kombinowane)"""
        results = {}

        for modality in ['face', 'signature', 'combined']:
            print(f"\n=== Trenowanie modeli dla: {modality.upper()} ===")
            features = features_dict[modality]

            if len(features) == 0:
                print(f"Brak danych dla {modality}")
                continue

            unique_labels = np.unique(labels)
            min_samples_per_class = min([np.sum(labels == label) for label in unique_labels])

            if min_samples_per_class < 3:
                print(f"UWAGA: Za mało próbek dla klasy ({min_samples_per_class}). Pomijam stratify.")
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42, stratify=labels
                )

            if modality == 'combined':
                scaler = self.scaler_combined
            elif modality == 'face':
                scaler = self.scaler_face
            else:
                scaler = self.scaler_signature

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train_scaled, y_train)
            knn_pred = knn.predict(X_test_scaled)
            knn_acc = accuracy_score(y_test, knn_pred) * 100

            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64) if modality == 'combined' else (64, 32),
                max_iter=2000,
                random_state=42,
                early_stopping=False
            )
            mlp.fit(X_train_scaled, y_train)
            mlp_pred = mlp.predict(X_test_scaled)
            mlp_acc = accuracy_score(y_test, mlp_pred) * 100

            if modality == 'combined':
                self.combined_knn, self.combined_mlp = knn, mlp
            elif modality == 'face':
                self.face_knn, self.face_mlp = knn, mlp
            else:
                self.signature_knn, self.signature_mlp = knn, mlp

            results[modality] = {
                'knn_acc': knn_acc,
                'mlp_acc': mlp_acc,
                'y_test': y_test,
                'mlp_pred': mlp_pred
            }

            print(f"[{modality.upper()}] k-NN: {knn_acc:.2f}%")
            print(f"[{modality.upper()}] MLP: {mlp_acc:.2f}%")
            print(f"\nRaport MLP - {modality}:")
            print(classification_report(y_test, mlp_pred, zero_division=0))

            cm = confusion_matrix(y_test, mlp_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Macierz pomyłek MLP - {modality}")
            plt.xlabel("Przewidziane")
            plt.ylabel("Rzeczywiste")
            plt.savefig(f"confusion_matrix_{modality}.png")
            plt.close()

        return results

    def score_level_fusion(self, face_path, signature_path, weights=(0.6, 0.4)):
        face_features = self.face_system.extract_face_features(face_path)
        signature_img = preprocess_image(signature_path)
        signature_features = extract_features(signature_img)

        if face_features is None:
            return "BRAK_ROZPOZNANIA", 0.0

        face_scaled = self.scaler_face.transform([face_features])
        sig_scaled = self.scaler_signature.transform([signature_features])

        face_pred = self.face_mlp.predict(face_scaled)[0]
        sig_pred = self.signature_mlp.predict(sig_scaled)[0]

        face_proba = self.face_mlp.predict_proba(face_scaled)[0]
        sig_proba = self.signature_mlp.predict_proba(sig_scaled)[0]

        pewnosc_face = np.max(face_proba)
        pewnosc_signature = np.max(sig_proba)

        pewnosc_laczna = weights[0] * pewnosc_face + weights[1] * pewnosc_signature

        if pewnosc_laczna <= 0.5:
            return "BRAK_ROZPOZNANIA", pewnosc_laczna

        if face_pred == sig_pred:
            return face_pred, pewnosc_laczna
        else:
            return "BRAK_ROZPOZNANIA", pewnosc_laczna

    def save_all_models(self, prefix="multimodal"):
        models = {
            f'{prefix}_combined_knn': self.combined_knn,
            f'{prefix}_combined_mlp': self.combined_mlp,
            f'{prefix}_face_knn': self.face_knn,
            f'{prefix}_face_mlp': self.face_mlp,
            f'{prefix}_signature_knn': self.signature_knn,
            f'{prefix}_signature_mlp': self.signature_mlp,
            f'{prefix}_scaler_combined': self.scaler_combined,
            f'{prefix}_scaler_face': self.scaler_face,
            f'{prefix}_scaler_signature': self.scaler_signature
        }

        for name, model in models.items():
            if model is not None:
                joblib.dump(model, f"{name}.pkl")

        print(f"Zapisano wszystkie modele z prefiksem: {prefix}")

    def load_models(self, prefix="multimodal"):
        try:
            self.combined_knn = joblib.load(f'{prefix}_combined_knn.pkl')
            self.combined_mlp = joblib.load(f'{prefix}_combined_mlp.pkl')
            self.face_knn = joblib.load(f'{prefix}_face_knn.pkl')
            self.face_mlp = joblib.load(f'{prefix}_face_mlp.pkl')
            self.signature_knn = joblib.load(f'{prefix}_signature_knn.pkl')
            self.signature_mlp = joblib.load(f'{prefix}_signature_mlp.pkl')
            self.scaler_combined = joblib.load(f'{prefix}_scaler_combined.pkl')
            self.scaler_face = joblib.load(f'{prefix}_scaler_face.pkl')
            self.scaler_signature = joblib.load(f'{prefix}_scaler_signature.pkl')
            print(f"Wczytano modele z prefiksem: {prefix}")
            return True
        except Exception as e:
            print(f"Błąd wczytywania modeli: {e}")
            return False


if __name__ == "__main__":
    multimodal_system = MultimodalBiometricSystem()

    face_dataset_path = "face_dataset"
    signature_dataset_path = "signature_dataset"

    features_dict, labels = multimodal_system.load_multimodal_dataset(
        face_dataset_path, signature_dataset_path
    )

    if len(labels) > 0:
        results = multimodal_system.train_all_models(features_dict, labels)

        print("\n" + "=" * 50)
        print("PODSUMOWANIE WYNIKÓW:")
        print("=" * 50)

        for modality in ['face', 'signature', 'combined']:
            if modality in results:
                print(f"{modality.upper():>12}: MLP {results[modality]['mlp_acc']:.2f}%")

        multimodal_system.save_all_models()

        print("\nSystem multimodalny gotowy do użycia!")
    else:
        print("Nie udało się wczytać danych!")
