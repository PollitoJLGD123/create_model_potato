from pathlib import Path
from keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from src.constant.constants import SEED

TYPECLASS = tuple[tf.data.Dataset, tf.data.Dataset, list]

def load_data(dataset_path: Path, img_size: int, batch_size: int) -> TYPECLASS:
  
  full_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='training',
    seed=SEED,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='categorical',
  )
  
  test_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='categorical',
  )
  
  class_names = full_dataset.class_names
  
  print(f"Nombres de las clases: {class_names}")
  print(f"Numero de clases encontradas: {len(class_names)}")
  
  # Verificar balance de clases en entrenamiento
  print("\n=== Distribución de clases en entrenamiento ===")
  class_counts = {}
  for batch_x, batch_y in full_dataset:
    for label in batch_y:
      class_idx = tf.argmax(label).numpy()
      class_name = class_names[class_idx]
      class_counts[class_name] = class_counts.get(class_name, 0) + 1
  
  total_train = sum(class_counts.values())
  for class_name, count in sorted(class_counts.items()):
    percentage = (count / total_train) * 100
    print(f"  {class_name}: {count} muestras ({percentage:.1f}%)")
  
  # Verificar balance en validación
  print("\n=== Distribución de clases en validación ===")
  val_class_counts = {}
  for batch_x, batch_y in test_dataset:
    for label in batch_y:
      class_idx = tf.argmax(label).numpy()
      class_name = class_names[class_idx]
      val_class_counts[class_name] = val_class_counts.get(class_name, 0) + 1
  
  total_val = sum(val_class_counts.values())
  for class_name, count in sorted(val_class_counts.items()):
    percentage = (count / total_val) * 100
    print(f"  {class_name}: {count} muestras ({percentage:.1f}%)")
  
  # Advertencia si hay desbalance significativo
  if total_train > 0:
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    if imbalance_ratio > 2.0:
      print(f"\n⚠️  ADVERTENCIA: Desbalance detectado (ratio máximo/mínimo: {imbalance_ratio:.2f})")
      print("   Considera usar class_weight en el entrenamiento o balancear el dataset.")
  
  return full_dataset, test_dataset, class_names