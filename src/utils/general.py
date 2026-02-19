import tensorflow as tf
from keras import layers
from src.constant.constants import AUTOTUNE
from keras import Sequential

def create_data_augmentation():
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    return data_augmentation

def prepare_dataset(dataset: tf.data.Dataset, data_augmentation=None, training=True):
    if training:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    
    # Normalizar imágenes
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=AUTOTUNE
    )
    
    # Optimizar rendimiento
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset