"""
Script pour inspecter le modèle TF-IDF et découvrir le bon mapping
"""

import numpy as np
from joblib import load

MODEL_PATH = "models/tfidf/tfidf_svm_best.pkl"

print("🔍 Inspection du modèle TF-IDF + SVM\n")

# Charger le modèle
model = load(MODEL_PATH)
print(f"✅ Modèle chargé: {type(model)}\n")

# Inspecter la structure
print("📊 Structure du modèle:")
if hasattr(model, "steps"):
    print("   Pipeline détecté avec les étapes:")
    for name, step in model.steps:
        print(f"      - {name}: {type(step).__name__}")

        # Si c'est le classifieur, vérifier les classes
        if name in ["classifier", "svm", "clf"] or "SVC" in type(step).__name__:
            if hasattr(step, "classes_"):
                print(f"\n   🏷️ Classes du modèle:")
                for i, cls in enumerate(step.classes_):
                    print(f"      {i}: {cls}")
print()

# Tester avec des exemples
test_cases = [
    "I need VPN access",
    "My laptop screen is broken",
    "Question about my vacation",
    "Order a new keyboard",
    "Need admin rights",
]

print("🧪 Tests de prédiction:\n")
for text in test_cases:
    pred = model.predict([text])[0]
    print(f"   Text: '{text}'")
    print(f"   → Prédiction brute: {pred} (type: {type(pred).__name__})")

    # Si c'est un tableau numpy
    if isinstance(pred, np.ndarray):
        print(f"   → Valeur: {pred.item()}")
    print()

# Vérifier les attributs du classifieur
print("🔧 Attributs du classifieur:")
for name, step in model.steps:
    if hasattr(step, "classes_"):
        print(f"   classes_: {step.classes_}")
    if hasattr(step, "n_classes_"):
        print(f"   n_classes_: {step.n_classes_}")

print("\n✨ Inspection terminée!")
