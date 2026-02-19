from keras import Model
import tensorflow as tf
from src.constant.constants import MODEL_PATH, METRICS_PATH, IMG_SIZE
import json

def evaluate_model_and_save(model: Model, val_dataset: tf.data.Dataset, class_names: list):
  
  ## evaluacion final del modelo
  print("Evaluacion final del modelo...")
  
  # return_dict=True evita depender del orden de las métricas
  results = model.evaluate(val_dataset, return_dict=True)
  loss = results["loss"]
  accuracy = results["accuracy"]
  precision = results["precision_1"]
  recall = results["recall_1"]
  
  print(f"Pérdida: {loss}")
  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  
  f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
  print(f"F1 Score: {f1_score}")
  
  # guardamos modelo final (crear carpeta si no existe)
  MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
  print("Guardando modelo final...")
  model.save(MODEL_PATH)
  print(f"Modelo guardado en {MODEL_PATH}")
  
  model_info = {
    'class_names': class_names,
    'img_size': IMG_SIZE,
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score)
    }
  }
  
  METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
  with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=4, ensure_ascii=False)
  print(f"Métricas guardadas en {METRICS_PATH}")