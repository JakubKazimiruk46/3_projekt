import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('TkAgg')


def visualize_pca(features, labels):
    """
    Wizualizacja danych po redukcji wymiarowości metodą PCA (Principal Component Analysis).

    Celem jest przedstawienie wektorów cech w przestrzeni 2D w taki sposób, by ocenić
    ich rozdzielczość i potencjalną separację między klasami (np. użytkownicy lub fałszerstwa).

    :param features: Wektory cech (np. Hu moments, liczba konturów itd.)
    :param labels: Etykiety klas (np. 'user01', 'genuine', 'forgery')
    """
    # Redukcja do 2 wymiarów – PCA znajduje główne składowe zmienności
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    # Wykres 2D – kolorowanie na podstawie etykiet
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2", s=50)
    plt.title("PCA - Wizualizacja podpisów")
    plt.savefig("pca_visualization.png")
    plt.close()


def visualize_tsne(features, labels):
    """
    Punkt dodatkowy: Wizualizacja danych po redukcji wymiarowości metodą t-SNE.

    t-SNE stara się zachować lokalne zależności (sąsiedztwa) i dobrze odwzorowuje nieliniowe relacje między próbkami.
    Nadaje się do wykrywania skupień, klastrów i oceny podobieństwa.

    :param features: Wektory cech
    :param labels: Etykiety klas
    """
    # Redukcja do 2D przy użyciu t-SNE – bardziej nieliniowa metoda niż PCA
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(features)

    # Wykres punktowy z kolorowaniem klas
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2", s=50)
    plt.title("t-SNE - Wizualizacja podpisów")
    plt.savefig("tsne_visualization.png")
    plt.close()
