from transformers import AutoModelForSequenceClassification

MODEL_PATH = "/content/drive/MyDrive/MLops/mlops_project/models/transformer"

# Chargement du modèle
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Ton vrai mapping des labels (dans le bon ordre de ton dataset d’entraînement)
id2label = {
    0: "Access",
    1: "Administrative rights",
    2: "HR Support",
    3: "Hardware",
    4: "Internal Project",
    5: "Miscellaneous",
    6: "Purchase",
    7: "Storage"
}
label2id = {v: k for k, v in id2label.items()}

# Injection dans la config du modèle
model.config.id2label = id2label
model.config.label2id = label2id

# Sauvegarde de la config mise à jour
model.save_pretrained(MODEL_PATH)

print("✅ Nouveau mapping sauvegardé avec succès dans le modèle !")
