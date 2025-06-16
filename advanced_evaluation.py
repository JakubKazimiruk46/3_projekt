import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def evaluate_fusion_system(multimodal_system, test_face_dir, test_signature_dir, weights_list=None):
    """Ewaluacja systemu fuzji zgodnie z metodyką zadania"""

    if weights_list is None:
        weights_list = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]

    print("=" * 60)
    print("EWALUACJA SYSTEMU FUZJI - METODYKA ZADANIA")
    print("=" * 60)

    # Przygotuj dane testowe
    test_data = prepare_test_data(test_face_dir, test_signature_dir)

    if len(test_data) == 0:
        print("❌ Brak danych testowych!")
        return

    print(f"Przygotowano {len(test_data)} par testowych")

    results = {}

    # Testuj różne kombinacje wag
    for w_face, w_sig in weights_list:
        print(f"\n📊 Testowanie wag: twarz={w_face}, podpis={w_sig}")

        predictions = []
        true_labels = []
        confidences = []
        rejection_count = 0
        class_mismatch_count = 0
        low_confidence_count = 0

        for face_path, sig_path, true_label in test_data:
            # Zastosuj metodykę fuzji z zadania
            prediction, confidence = multimodal_system.score_level_fusion(
                face_path, sig_path, weights=(w_face, w_sig)
            )

            predictions.append(prediction)
            true_labels.append(true_label)
            confidences.append(confidence)

            # Statystyki odrzuceń
            if prediction == "BRAK_ROZPOZNANIA":
                rejection_count += 1
                if confidence <= 0.5:
                    low_confidence_count += 1
                else:
                    class_mismatch_count += 1

        # Oblicz metryki
        metrics = calculate_fusion_metrics(predictions, true_labels, confidences)
        metrics['rejection_rate'] = rejection_count / len(test_data) * 100
        metrics['class_mismatch_rate'] = class_mismatch_count / len(test_data) * 100
        metrics['low_confidence_rate'] = low_confidence_count / len(test_data) * 100

        results[(w_face, w_sig)] = metrics

        # Wyświetl wyniki
        print(f"   Dokładność (bez odrzuceń): {metrics['accuracy_no_reject']:.2f}%")
        print(f"   Dokładność (z odrzuceniami): {metrics['accuracy_with_reject']:.2f}%")
        print(f"   Współczynnik odrzuceń: {metrics['rejection_rate']:.2f}%")
        print(f"     - niska pewność (≤0.5): {metrics['low_confidence_rate']:.2f}%")
        print(f"     - różne klasy: {metrics['class_mismatch_rate']:.2f}%")
        print(f"   Średnia pewność: {metrics['avg_confidence']:.3f}")

    # Porównanie wyników
    print_fusion_comparison(results)
    plot_fusion_results(results)

    return results


def prepare_test_data(face_dir, signature_dir):
    """Przygotuj pary testowe (twarz, podpis, etykieta)"""
    test_data = []

    # Znajdź wspólnych użytkowników
    face_users = set(os.listdir(face_dir)) if os.path.exists(face_dir) else set()
    sig_users = set(os.listdir(signature_dir)) if os.path.exists(signature_dir) else set()
    common_users = face_users.intersection(sig_users)

    for user in common_users:
        face_user_dir = os.path.join(face_dir, user)
        sig_user_dir = os.path.join(signature_dir, user)

        if not (os.path.isdir(face_user_dir) and os.path.isdir(sig_user_dir)):
            continue

        face_files = [f for f in os.listdir(face_user_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        sig_files = [f for f in os.listdir(sig_user_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Sparuj pliki
        min_files = min(len(face_files), len(sig_files))

        for i in range(min_files):
            face_path = os.path.join(face_user_dir, face_files[i])
            sig_path = os.path.join(sig_user_dir, sig_files[i])
            test_data.append((face_path, sig_path, user))

    return test_data


def calculate_fusion_metrics(predictions, true_labels, confidences):
    """Oblicz metryki dla systemu fuzji"""

    # Usuń przypadki z brakiem rozpoznania dla dokładności bez odrzuceń
    valid_indices = [i for i, pred in enumerate(predictions) if pred != "BRAK_ROZPOZNANIA"]

    if len(valid_indices) > 0:
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_labels = [true_labels[i] for i in valid_indices]
        accuracy_no_reject = accuracy_score(valid_true_labels, valid_predictions) * 100
    else:
        accuracy_no_reject = 0.0

    # Dokładność z odrzuceniami (odrzucenia liczone jako błąd)
    modified_predictions = [pred if pred != "BRAK_ROZPOZNANIA" else "WRONG" for pred in predictions]
    accuracy_with_reject = accuracy_score(true_labels, modified_predictions) * 100

    return {
        'accuracy_no_reject': accuracy_no_reject,
        'accuracy_with_reject': accuracy_with_reject,
        'avg_confidence': np.mean(confidences),
        'valid_predictions': len(valid_indices),
        'total_predictions': len(predictions)
    }


def print_fusion_comparison(results):
    """Wyświetl porównanie różnych wag"""
    print(f"\n{'=' * 60}")
    print("PORÓWNANIE STRATEGII FUZJI")
    print(f"{'=' * 60}")

    print(f"{'Wagi (T,P)':<12} {'Dok.bez odrzuć':<13} {'Dok.z odrzuć':<12} {'Odrzucenia':<10} {'Śred.pewność':<12}")
    print("-" * 60)

    for (w_face, w_sig), metrics in results.items():
        print(f"({w_face:.1f},{w_sig:.1f}){'':<4} "
              f"{metrics['accuracy_no_reject']:>8.1f}%{'':<4} "
              f"{metrics['accuracy_with_reject']:>7.1f}%{'':<4} "
              f"{metrics['rejection_rate']:>6.1f}%{'':<3} "
              f"{metrics['avg_confidence']:>8.3f}")

    # Najlepsza strategia
    best_weights = max(results.keys(), key=lambda w: results[w]['accuracy_no_reject'])
    best_acc = results[best_weights]['accuracy_no_reject']

    print(f"\n🏆 NAJLEPSZA STRATEGIA: Wagi {best_weights} (Dokładność: {best_acc:.1f}%)")


def plot_fusion_results(results):
    """Wykres wyników fuzji"""
    weights_labels = [f"({w[0]:.1f},{w[1]:.1f})" for w in results.keys()]
    acc_no_reject = [results[w]['accuracy_no_reject'] for w in results.keys()]
    acc_with_reject = [results[w]['accuracy_with_reject'] for w in results.keys()]
    rejection_rates = [results[w]['rejection_rate'] for w in results.keys()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Wykres dokładności
    x = np.arange(len(weights_labels))
    width = 0.35

    ax1.bar(x - width / 2, acc_no_reject, width, label='Bez odrzuceń', alpha=0.8, color='skyblue')
    ax1.bar(x + width / 2, acc_with_reject, width, label='Z odrzuceniami', alpha=0.8, color='lightcoral')

    ax1.set_xlabel('Wagi (Twarz, Podpis)')
    ax1.set_ylabel('Dokładność (%)')
    ax1.set_title('Dokładność vs Strategia fuzji')
    ax1.set_xticks(x)
    ax1.set_xticklabels(weights_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Wykres odrzuceń
    ax2.bar(weights_labels, rejection_rates, alpha=0.8, color='orange')
    ax2.set_xlabel('Wagi (Twarz, Podpis)')
    ax2.set_ylabel('Współczynnik odrzuceń (%)')
    ax2.set_title('Współczynnik odrzuceń vs Strategia fuzji')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fusion_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("📊 Wykresy zapisane w: fusion_evaluation_results.png")


def detailed_analysis(multimodal_system, test_face_dir, test_signature_dir):
    """Szczegółowa analiza przypadków brzegowych"""
    print(f"\n{'=' * 60}")
    print("SZCZEGÓŁOWA ANALIZA PRZYPADKÓW")
    print(f"{'=' * 60}")

    test_data = prepare_test_data(test_face_dir, test_signature_dir)

    if len(test_data) == 0:
        return

    # Analiza dla optymalnych wag
    optimal_weights = (0.6, 0.4)  # Można dostosować na podstawie wyników

    print(f"Analiza dla wag: {optimal_weights}")

    cases = {
        'correct_high_conf': [],
        'correct_low_conf': [],
        'rejected_low_conf': [],
        'rejected_mismatch': [],
        'incorrect': []
    }

    for face_path, sig_path, true_label in test_data[:20]:  # Pierwsze 20 przypadków
        prediction, confidence = multimodal_system.score_level_fusion(
            face_path, sig_path, weights=optimal_weights
        )

        # Klasyfikuj przypadek
        if prediction == true_label:
            if confidence > 0.7:
                cases['correct_high_conf'].append((true_label, confidence))
            else:
                cases['correct_low_conf'].append((true_label, confidence))
        elif prediction == "BRAK_ROZPOZNANIA":
            if confidence <= 0.5:
                cases['rejected_low_conf'].append((true_label, confidence))
            else:
                cases['rejected_mismatch'].append((true_label, confidence))
        else:
            cases['incorrect'].append((true_label, prediction, confidence))

    # Wyświetl analizę
    print(f"\n📈 Poprawne rozpoznania z wysoką pewnością: {len(cases['correct_high_conf'])}")
    if cases['correct_high_conf']:
        avg_conf = np.mean([c[1] for c in cases['correct_high_conf']])
        print(f"   Średnia pewność: {avg_conf:.3f}")

    print(f"\n📉 Poprawne rozpoznania z niską pewnością: {len(cases['correct_low_conf'])}")
    if cases['correct_low_conf']:
        avg_conf = np.mean([c[1] for c in cases['correct_low_conf']])
        print(f"   Średnia pewność: {avg_conf:.3f}")

    print(f"\n❌ Odrzucenia (niska pewność): {len(cases['rejected_low_conf'])}")
    print(f"❌ Odrzucenia (różne klasy): {len(cases['rejected_mismatch'])}")
    print(f"🚫 Niepoprawne rozpoznania: {len(cases['incorrect'])}")

    if cases['incorrect']:
        print("   Przykłady błędów:")
        for true_label, pred, conf in cases['incorrect'][:3]:
            print(f"     {true_label} → {pred} (pewność: {conf:.3f})")


# Główna funkcja uruchamiająca ewaluację
def run_fusion_evaluation():
    """Uruchom pełną ewaluację systemu fuzji"""
    from multimodal_biometric import MultimodalBiometricSystem

    # Inicjalizacja
    multimodal_system = MultimodalBiometricSystem()

    # Sprawdź czy modele istnieją
    if not multimodal_system.load_models("multimodal"):
        print("❌ Brak wytrenowanych modeli! Uruchom najpierw trenowanie.")
        return

    # Ścieżki do danych testowych
    test_face_dir = "face_dataset"
    test_signature_dir = "signature_dataset"

    # Ewaluacja różnych strategii fuzji
    results = evaluate_fusion_system(
        multimodal_system, test_face_dir, test_signature_dir
    )

    # Szczegółowa analiza
    detailed_analysis(multimodal_system, test_face_dir, test_signature_dir)

    print(f"\n🎯 EWALUACJA FUZJI ZAKOŃCZONA!")


if __name__ == "__main__":
    run_fusion_evaluation()
