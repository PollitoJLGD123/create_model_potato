import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.metrics import Precision, Recall, CategoricalAccuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam

from src.load.load_data import load_data
from src.constant.constants import (
    DATASET_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    SAMPLE_PATH,
)
from src.utils.general import create_data_augmentation, prepare_dataset
from src.generator.mobilevit import create_model_mobilevit


# Rutas específicas para el modelo MobileViT
MODEL_MOBILEVIT_PATH = Path(__file__).parent / "model" / "model_mobilevit.keras"
HISTORY_MOBILEVIT_PATH = SAMPLE_PATH / "history_mobilevit.png"
METRICS_MOBILEVIT_PATH = SAMPLE_PATH / "metrics_mobilevit.json"


def train_model_mobilevit(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
):
    """
    Entrena el modelo MobileViT con los mismos hiperparámetros
    que el flujo basado en EfficientNet (LR, épocas, callbacks, métricas).
    """
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(), Recall(), CategoricalAccuracy()],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath=str(MODEL_MOBILEVIT_PATH),
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    return model, history.history


def evaluate_model_and_save_mobilevit(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list,
):
    """
    Evalúa el modelo en el conjunto de test y guarda
    modelo + métricas en rutas específicas de MobileViT.
    """
    print("Evaluacion final del modelo MobileViT...")

    results = model.evaluate(test_dataset, return_dict=True)
    loss = results.get("loss")
    accuracy = results.get("accuracy")

    # Los nombres de las métricas pueden variar; buscamos por prefijo
    precision_key = next(
        (k for k in results.keys() if k.startswith("precision")), None
    )
    recall_key = next(
        (k for k in results.keys() if k.startswith("recall")), None
    )

    precision = results.get(precision_key) if precision_key else None
    recall = results.get(recall_key) if recall_key else None

    print(f"Pérdida: {loss}")
    print(f"Accuracy: {accuracy}")
    if precision is not None:
        print(f"Precision: {precision}")
    if recall is not None:
        print(f"Recall: {recall}")

    if precision is not None and recall is not None:
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        print(f"F1 Score: {f1_score}")
    else:
        f1_score = None

    # Guardar modelo final
    MODEL_MOBILEVIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Guardando modelo MobileViT final...")
    model.save(MODEL_MOBILEVIT_PATH)
    print(f"Modelo MobileViT guardado en {MODEL_MOBILEVIT_PATH}")

    metrics = {
        "accuracy": float(accuracy) if accuracy is not None else None,
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "f1_score": float(f1_score) if f1_score is not None else None,
    }

    model_info = {
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "metrics": metrics,
    }

    METRICS_MOBILEVIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_MOBILEVIT_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    print(f"Métricas MobileViT guardadas en {METRICS_MOBILEVIT_PATH}")


def generate_history_mobilevit(history: dict):
    """
    Genera la gráfica de historial de entrenamiento para MobileViT
    y la guarda en HISTORY_MOBILEVIT_PATH.
    """
    required_keys = ("loss", "val_loss", "accuracy", "val_accuracy")
    missing = [k for k in required_keys if k not in history]
    if missing:
        print(
            f"Advertencia: no se puede graficar el historial MobileViT "
            f"(faltan: {missing}). Se omite."
        )
        return

    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history["loss"], label="Entrenamiento")
    axes[0].plot(history["val_loss"], label="Validación")
    axes[0].set_title("Pérdida durante el entrenamiento (MobileViT)")
    axes[0].set_xlabel("Épocas")
    axes[0].set_ylabel("Pérdida")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["accuracy"], label="Entrenamiento")
    axes[1].plot(history["val_accuracy"], label="Validación")
    axes[1].set_title("Precisión durante el entrenamiento (MobileViT)")
    axes[1].set_xlabel("Épocas")
    axes[1].set_ylabel("Precisión")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    HISTORY_MOBILEVIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(HISTORY_MOBILEVIT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Gráfica de historial MobileViT guardada en {HISTORY_MOBILEVIT_PATH}")


def main():
    print("=" * 60)
    print("INICIAMOS ENTRENAMIENTO DE MODELO MobileViT")
    print("=" * 60)

    # Cargar datos
    print("Cargando datos...")
    train_dataset, val_dataset, test_dataset, class_names = load_data(
        DATASET_PATH, IMG_SIZE, BATCH_SIZE
    )

    # Preparar datasets
    print("Preparando dataset...")
    data_augmentation = create_data_augmentation()
    train_dataset = prepare_dataset(
        train_dataset, data_augmentation, training=True
    )
    val_dataset = prepare_dataset(val_dataset, training=False)
    test_dataset = prepare_dataset(test_dataset, training=False)

    # Crear modelo
    print("Creando modelo MobileViT...")
    model = create_model_mobilevit(len(class_names), IMG_SIZE)
    model.summary()

    # Entrenar modelo
    print("Entrenando modelo MobileViT...")
    model, history = train_model_mobilevit(model, train_dataset, val_dataset)

    # Evaluar y guardar
    print("Evaluando y guardando modelo MobileViT...")
    evaluate_model_and_save_mobilevit(model, test_dataset, class_names)

    # Generar historial
    print("Generando gráfica de historial MobileViT...")
    generate_history_mobilevit(history)

    print("=" * 60)
    print("FIN DEL ENTRENAMIENTO DE MODELO MobileViT")
    print("=" * 60)


if __name__ == "__main__":
    main()

