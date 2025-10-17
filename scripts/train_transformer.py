import pandas as pd
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch
from datasets import Dataset
import numpy as np
import os

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

def train_transformer(input_path, output_model_path):
    with mlflow.start_run():
        # Charger les données
        df = pd.read_csv(input_path)
        X = df['Document']
        y = df['Topic_group']
        # Encoder les labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        # Créer un dataset Hugging Face
        train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
        test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})
        # Charger le tokenizer et le modèle
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=len(le.classes_))
        # Tokenisation
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        # Configurer les arguments d'entraînement
        training_args = TrainingArguments(
            output_dir=output_model_path,
            eval_strategy='epoch',  # Changé de evaluation_strategy
            save_strategy='epoch',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            report_to='none',
        )
        # Configurer le Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        # Entraîner
        trainer.train()
        # Évaluer
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("output_model_path", output_model_path)
        mlflow.log_param("model_name", "distilbert-base-multilingual-cased")
        mlflow.log_param("num_epochs", training_args.num_train_epochs)
        # Sauvegarder le modèle
        os.makedirs(output_model_path, exist_ok=True)
        trainer.save_model(output_model_path)
        tokenizer.save_pretrained(output_model_path)
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model"
        )
        print(f"Model trained and saved to {output_model_path}")

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///content/drive/MyDrive/MLops/mlruns")
    mlflow.set_experiment("mlops_project_experiment")
    input_path = "data/processed/tickets_processed.csv"
    output_model_path = "models/transformer"
    train_transformer(input_path, output_model_path)
