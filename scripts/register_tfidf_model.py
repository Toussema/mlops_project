"""
Script pour enregistrer le modèle TF-IDF + SVM dans MLflow Registry
"""

import os

import mlflow
from joblib import load
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = "file:///C:/Users/touha/Desktop/mlops_project/mlruns"
MODEL_NAME = "TfidfSvmTicketClassifier"
MODEL_PATH = "models/tfidf/tfidf_svm_best.pkl"


def register_tfidf_model():
    """Enregistre le modèle TF-IDF + SVM dans MLflow"""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("production_models")

    print("🚀 Enregistrement du modèle TF-IDF + SVM\n")

    # Charger le modèle
    print(f"📂 Chargement depuis: {MODEL_PATH}")
    model = load(MODEL_PATH)
    print("✅ Modèle chargé avec joblib\n")

    # Logger dans MLflow
    with mlflow.start_run(run_name="tfidf_svm_production") as run:

        # Logger les paramètres
        mlflow.log_param("model_type", "tfidf_svm")
        mlflow.log_param("algorithm", "SVM with TF-IDF vectorization")

        # Logger les métriques (remplacez par les vraies valeurs)
        mlflow.log_metric("eval_accuracy", 0.90)  # Voir classification_report_best.json
        mlflow.log_metric("eval_f1", 0.89)

        # Logger le modèle avec sklearn
        print("📦 Logging du modèle dans MLflow...")
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

        run_id = run.info.run_id
        print(f"✅ Run créé: {run_id}\n")

    # Promouvoir en Production
    client = MlflowClient()

    # Récupérer la dernière version
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if latest_versions:
        version = latest_versions[0].version

        # Promouvoir
        try:
            client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=version)
            print(f"✅ Alias 'champion' attribué à la version {version}")
        except AttributeError:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=version, stage="Production"
            )
            print(f"✅ Modèle promu en Production")

    print(f"\n🎉 Enregistrement terminé!")
    print(f"📊 Modèle: {MODEL_NAME}")
    print(f"👉 MLflow UI: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    try:
        register_tfidf_model()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
