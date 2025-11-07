"""
Script pour enregistrer le modèle Transformer dans MLflow Registry
et le promouvoir en Production
"""

import os
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = "file:///C:/Users/touha/Desktop/mlops_project/mlruns"
EXPERIMENT_NAME = "mlops_project_experiment"
MODEL_NAME = "TransformerTicketClassifier"


def register_best_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"❌ Expérience '{EXPERIMENT_NAME}' introuvable")
        return

    print(f"✅ Expérience trouvée: {experiment.experiment_id}")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        print("❌ Aucun run trouvé")
        return

    # Filtrer uniquement les runs avec un modèle enregistré
    valid_runs = []
    for _, run in runs.iterrows():
        run_id = run.run_id
        artifact_path = (
            f"C:/Users/touha/Desktop/mlops_project/mlruns/"
            f"{experiment.experiment_id}/{run_id}/artifacts/model"
        )
        if os.path.exists(artifact_path):
            valid_runs.append(run)

    if not valid_runs:
        print("❌ Aucun run ne contient de modèle logué sous 'artifacts/model'")
        return

    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    metric_name = (
        "metrics.eval_accuracy" if "metrics.eval_accuracy" in metric_cols else "metrics.eval_f1"
    )

    runs = runs.sort_values(by=metric_name, ascending=False)
    best_run = None
    for _, run in runs.iterrows():
        run_id = run.run_id
        path = (
            f"C:/Users/touha/Desktop/mlops_project/mlruns/"
            f"{experiment.experiment_id}/{run_id}/artifacts/model"
        )
        if os.path.exists(path):
            best_run = run
            break

    if best_run is None:
        print("❌ Aucun modèle valide trouvé.")
        return

    run_id = best_run.run_id
    metric_value = best_run[metric_name]

    print("\n📈 Meilleur run trouvé avec modèle:")
    print(f"   Run ID: {run_id}")
    print(f"   {metric_name}: {metric_value:.4f}")

    # Enregistrer et promouvoir
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=model_version.version,
        key="deployment_date",
        value=datetime.now().strftime("%Y-%m-%d"),
    )

    print(f"\n🚀 Modèle {MODEL_NAME} promu en Production (version {model_version.version})")


if __name__ == "__main__":
    register_best_model()
