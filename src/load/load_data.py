from pathlib import Path
from keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from src.constant.constants import SEED

TYPECLASS = tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list]


def load_data(dataset_path: Path, img_size: int, batch_size: int) -> TYPECLASS:
  """
  Espera una estructura:
    dataset_path/
      train/
        class_1/...
        class_2/...
      valid/
        class_1/...
        class_2/...
      test/
        class_1/...
        class_2/...
  """
  train_dir = dataset_path / "train"
  valid_dir = dataset_path / "valid"
  test_dir = dataset_path / "test"

  train_dataset = image_dataset_from_directory(
    train_dir,
    seed=SEED,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical",
  )

  val_dataset = image_dataset_from_directory(
    valid_dir,
    seed=SEED,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical",
  )

  test_dataset = image_dataset_from_directory(
    test_dir,
    seed=SEED,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False,
  )

  class_names = train_dataset.class_names

  print(f"Nombres de las clases: {class_names}")
  print(f"Numero de clases encontradas: {len(class_names)}")

  # Verificar balance de clases en entrenamiento
  print("\n=== Distribución de clases en entrenamiento (train) ===")
  train_class_counts = {}
  for _, batch_y in train_dataset:
    for label in batch_y:
      class_idx = tf.argmax(label).numpy()
      class_name = class_names[class_idx]
      train_class_counts[class_name] = train_class_counts.get(class_name, 0) + 1

  total_train = sum(train_class_counts.values())
  for class_name, count in sorted(train_class_counts.items()):
    percentage = (count / total_train) * 100 if total_train > 0 else 0.0
    print(f"  {class_name}: {count} muestras ({percentage:.1f}%)")

  # Verificar balance en validación
  print("\n=== Distribución de clases en validación (valid) ===")
  val_class_counts = {}
  for _, batch_y in val_dataset:
    for label in batch_y:
      class_idx = tf.argmax(label).numpy()
      class_name = class_names[class_idx]
      val_class_counts[class_name] = val_class_counts.get(class_name, 0) + 1

  total_val = sum(val_class_counts.values())
  for class_name, count in sorted(val_class_counts.items()):
    percentage = (count / total_val) * 100 if total_val > 0 else 0.0
    print(f"  {class_name}: {count} muestras ({percentage:.1f}%)")

  # Verificar balance en test
  print("\n=== Distribución de clases en test ===")
  test_class_counts = {}
  for _, batch_y in test_dataset:
    for label in batch_y:
      class_idx = tf.argmax(label).numpy()
      class_name = class_names[class_idx]
      test_class_counts[class_name] = test_class_counts.get(class_name, 0) + 1

  total_test = sum(test_class_counts.values())
  for class_name, count in sorted(test_class_counts.items()):
    percentage = (count / total_test) * 100 if total_test > 0 else 0.0
    print(f"  {class_name}: {count} muestras ({percentage:.1f}%)")

  # Advertencia si hay desbalance significativo en train
  if total_train > 0:
    max_count = max(train_class_counts.values())
    min_count = min(train_class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    if imbalance_ratio > 2.0:
      print(f"\n⚠️  ADVERTENCIA: Desbalance en train (ratio máximo/mínimo: {imbalance_ratio:.2f})")
      print("   Considera usar class_weight en el entrenamiento o balancear el dataset.")

  return train_dataset, val_dataset, test_dataset, class_names