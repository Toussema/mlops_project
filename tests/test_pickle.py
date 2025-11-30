"""
Script pour tester si le fichier pickle est valide
"""

import os
import pickle

# Chemins à tester
paths_to_test = [
    "models/tfidf/tfidf_svm_best.pkl",
    "C:/Users/touha/Downloads/MLops/MLops/models/tfidf_svm_best.pkl",
]

for path in paths_to_test:
    print(f"\n{'='*60}")
    print(f"Test: {path}")
    print("=" * 60)

    if not os.path.exists(path):
        print(f"❌ Fichier introuvable: {path}")
        continue

    # Taille du fichier
    size = os.path.getsize(path)
    print(f"📊 Taille: {size:,} bytes ({size/1024/1024:.2f} MB)")

    # Lire les premiers bytes
    try:
        with open(path, "rb") as f:
            first_bytes = f.read(10)
            print(f"🔍 Premiers bytes: {first_bytes.hex()}")

            # Vérifier si c'est un pickle
            if first_bytes[0:2] == b"\x80\x04":
                print("✅ Format pickle détecté (Protocol 4)")
            elif first_bytes[0:2] == b"\x80\x03":
                print("✅ Format pickle détecté (Protocol 3)")
            elif first_bytes[0:2] == b"\x80\x05":
                print("✅ Format pickle détecté (Protocol 5)")
            else:
                print(f"⚠️ Format inhabituel: {first_bytes[0:2].hex()}")
    except Exception as e:
        print(f"❌ Erreur lecture: {e}")
        continue

    # Essayer de charger
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Chargement réussi!")
        print(f"   Type: {type(model)}")

        # Si c'est un pipeline sklearn, afficher les étapes
        if hasattr(model, "steps"):
            print(f"   Pipeline steps:")
            for name, step in model.steps:
                print(f"      - {name}: {type(step).__name__}")

        # Tester une prédiction
        try:
            test_text = ["I need access to the database"]
            prediction = model.predict(test_text)
            print(f"   Test prédiction: {prediction}")
        except Exception as e:
            print(f"   ⚠️ Prédiction échouée: {e}")

    except Exception as e:
        print(f"❌ Chargement échoué: {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'='*60}")
print("✨ Test terminé")
print("=" * 60)
