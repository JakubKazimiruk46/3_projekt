import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib

matplotlib.use('TkAgg')


# Punkt e: Implementacja klasyfikatora k-NN oraz MLP
def train_and_evaluate(features, labels, model_name="multi"):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42)

    scaler = StandardScaler()  # normalizacja cech
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Klasyfikator k-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test_scaled)) * 100

    # Klasyfikator MLP (sieć neuronowa)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test_scaled)) * 100

    # Raport klasyfikacji
    print(f"\n[{model_name}] Dokładność k-NN: {knn_acc:.2f}%")
    print(f"[{model_name}] Dokładność MLP:  {mlp_acc:.2f}%\n")
    y_pred = mlp.predict(X_test_scaled)
    print(f"Raport MLP - {model_name}:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Macierz pomyłek MLP - {model_name}")
    plt.xlabel("Przewidziane")
    plt.ylabel("Rzeczywiste")
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.close()

    # Zapis modelu i skalera
    joblib.dump(mlp, f"mlp_model_{model_name}.pkl")
    joblib.dump(scaler, f"scaler_{model_name}.pkl")
    print(f"Zapisano modele do: mlp_model_{model_name}.pkl i scaler_{model_name}.pkl")
