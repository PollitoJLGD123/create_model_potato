from pathlib import Path
import tensorflow as tf

# nombre del modelo entrenado
MODEL_NAME = "model_efficient.keras"

# ruta del modelo entrenado
MODEL_PATH = Path(__file__).parent.parent.parent / "model" / MODEL_NAME

# ruta del dataset (raiz de la carpeta data)
DATASET_PATH = Path(__file__).parent.parent.parent / "data"

# PARAMETROS DE ENTRENAMIENTO
BATCH_SIZE = 32
IMG_SIZE = 224  # EfficientNet usa 224x224 por defecto
LEARNING_RATE = 0.001
SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 10

#### SAMPLES
SAMPLE_PATH = Path(__file__).parent.parent.parent / "samples"

## graficas
HISTORY_NAME = "history.png"
HISTORY_PATH = SAMPLE_PATH / HISTORY_NAME

## métricas adicionales
METRICS_NAME = "metrics.json"
METRICS_PATH = SAMPLE_PATH / METRICS_NAME