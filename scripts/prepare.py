import pandas as pd
import os
import mlflow
import re

def preprocess_data(input_path, output_path):
    with mlflow.start_run():
        # Charger le dataset
        df = pd.read_csv(input_path)
        # Supprimer les lignes avec des valeurs manquantes
        df = df.dropna()
        # Nettoyer la colonne Document
        df['Document'] = df['Document'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        # Enregistrer le dataset prétraité
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        # Logger les paramètres et métriques dans MLflow
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("output_path", output_path)
        mlflow.log_metric("num_rows", len(df))
        mlflow.log_metric("num_columns", len(df.columns))
        print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///content/drive/MyDrive/MLops/mlruns")
    mlflow.set_experiment("mlops_project_experiment")
    input_path = "data/raw/tickets.csv"
    output_path = "data/processed/tickets_processed.csv"
    preprocess_data(input_path, output_path)
