import os
from .load.load_data import load_data
from .constant.constants import DATASET_PATH, IMG_SIZE, BATCH_SIZE
from .utils.general import create_data_augmentation
from .utils.general import prepare_dataset
from .generator.efficient import create_model_efficient
from .train.train_model import train_model
from .evaluate.evaluate_save import evaluate_model_and_save
from .history.generate_history import generate_history

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
  print("=" * 60)
  print("INICIAMOS ENTRENAMIENTO DE MODELO")
  print("=" * 60)
  
  ## cargamos datos
  print("Cargando datos...")
  train_dataset, test_dataset, class_names = load_data(DATASET_PATH, IMG_SIZE, BATCH_SIZE)
  
  ## preparamos dataset
  print("Preparando dataset...")
  data_augmentation = create_data_augmentation()
  train_dataset = prepare_dataset(train_dataset, data_augmentation, training=True)
  test_dataset = prepare_dataset(test_dataset, training=False)
  
  # creamos el modelo
  print("Creando modelo...")
  model, base_model = create_model_efficient(len(class_names), IMG_SIZE)
  model.summary() ## mostramos el modelo
  
  # entrenamos el modelo
  print("Entrenando modelo...")
  model, history = train_model(model, base_model, train_dataset, test_dataset)
  
  ## evaluamos y guardamos el modelo
  print("Evaluando y guardando modelo...")
  evaluate_model_and_save(model, test_dataset, class_names)
  
  ## generamos grafica de historial
  print("Generando grafica de historial...")
  generate_history(history)
  
  print("=" * 60)
  print("FIN DEL ENTRENAMIENTO DE MODELO")
  print("=" * 60)
  