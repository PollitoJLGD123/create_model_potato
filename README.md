# Entrenamiento de modelo de clasificación (EfficientNet)

Proyecto para entrenar un modelo de clasificación de imágenes con **EfficientNetB0** (transfer learning). Incluye dos fases: entrenar solo el cabezal y luego fine-tuning parcial del backbone.

## Requisitos

- Python 3.x
- Dependencias en `requirements.txt`

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura del proyecto

```
create_model/
├── main.py                 # Punto de entrada
├── requirements.txt
├── data/                   # Dataset (ver abajo)
├── model/                  # Modelo guardado (model.keras)
├── samples/                # Gráficas y métricas (history.png, metrics.json)
└── src/
    ├── __init__.py         # Flujo principal (main)
    ├── constant/
    │   └── constants.py    # Rutas, epochs, batch size, etc.
    ├── load/
    │   └── load_data.py    # Carga dataset desde carpeta
    ├── utils/
    │   └── general.py      # Data augmentation y preparación del dataset
    ├── generator/
    │   └── efficient.py   # Definición del modelo EfficientNetB0 + cabezal
    ├── train/
    │   └── train_model.py # Entrenamiento en dos fases
    ├── evaluate/
    │   └── evaluate_save.py # Evaluación, guardado del modelo y métricas
    └── history/
        └── generate_history.py # Gráficas de pérdida y precisión
```

## Dataset

Coloca tus imágenes en la carpeta **`data/`** con una subcarpeta por clase:

```
data/
├── clase_1/
│   ├── img1.jpg
│   └── ...
├── clase_2/
│   └── ...
└── clase_3/
    └── ...
```

El script usa `image_dataset_from_directory` con **80% entrenamiento** y **20% validación**.

## Cómo ejecutar

Desde la raíz del proyecto:

```bash
python main.py
```

El flujo hace: carga de datos → aumento de datos → creación del modelo → entrenamiento (2 fases) → evaluación → guardado del modelo en `model/model.keras` → guardado de métricas en `samples/metrics.json` → gráfica de historial en `samples/history.png`.

## Configuración

En `src/constant/constants.py` puedes ajustar:

- `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`
- `IMG_SIZE` (por defecto 224 para EfficientNet)
- `DATASET_PATH`, `MODEL_PATH`, `SAMPLE_PATH` si quieres otras rutas
