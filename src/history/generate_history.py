import matplotlib.pyplot as plt
from src.constant.constants import HISTORY_PATH

_REQUIRED_KEYS = ("loss", "val_loss", "accuracy", "val_accuracy")

def generate_history(history: dict):
  missing = [k for k in _REQUIRED_KEYS if k not in history]
  if missing:
    print(f"Advertencia: no se puede graficar el historial (faltan: {missing}). Se omite.")
    return

  _, axes = plt.subplots(1, 2, figsize=(15, 5))

  axes[0].plot(history["loss"], label="Entrenamiento")
  axes[0].plot(history["val_loss"], label="Validación")
  axes[0].set_title("Pérdida durante el entrenamiento")
  axes[0].set_xlabel("Épocas")
  axes[0].set_ylabel("Pérdida")
  axes[0].legend()
  axes[0].grid(True)

  axes[1].plot(history["accuracy"], label="Entrenamiento")
  axes[1].plot(history["val_accuracy"], label="Validación")
  axes[1].set_title("Precisión durante el entrenamiento")
  axes[1].set_xlabel("Épocas")
  axes[1].set_ylabel("Precisión")
  axes[1].legend()
  axes[1].grid(True)

  plt.tight_layout()

  HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(HISTORY_PATH, dpi=300, bbox_inches="tight")
  plt.show()

  print(f"Gráfica de historial guardada en {HISTORY_PATH}")