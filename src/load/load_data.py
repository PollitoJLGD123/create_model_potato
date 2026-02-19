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
  
  return full_dataset, test_dataset, class_names