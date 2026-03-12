import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import redis
import streamlit as st
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input

from src.constant.constants import MODEL_PATH, METRICS_PATH, HISTORY_PATH, IMG_SIZE, SAMPLE_PATH


# Rutas específicas para MobileViT y carpeta pública para imágenes analizadas
MOBILEVIT_MODEL_PATH = Path(__file__).parent / "model" / "model_mobilevit.keras"
MOBILEVIT_METRICS_PATH = SAMPLE_PATH / "metrics_mobilevit.json"
MOBILEVIT_HISTORY_PATH = SAMPLE_PATH / "history_mobilevit.png"

PUBLIC_DIR = Path(__file__).parent / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

REDIS_KEY_PREDICTIONS = "thorax_model_predictions"
REDIS_TTL_SECONDS = 60 * 60 * 12  # 12 horas


st.set_page_config(
    page_title="Dashboard modelo cáncer de tórax",
    layout="wide",
)


@st.cache_resource
def get_redis_client() -> redis.Redis:
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


@st.cache_resource
def load_efficient_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_mobilevit_model():
    return load_model(MOBILEVIT_MODEL_PATH)


@st.cache_data
def load_efficient_info():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info


@st.cache_data
def load_mobilevit_info():
    with open(MOBILEVIT_METRICS_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info


def preprocess_image(file, img_size: int):
    image = Image.open(file).convert("RGB")
    image = image.resize((img_size, img_size))

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array, image


def save_prediction_to_redis(entry: Dict[str, Any]) -> None:
    client = get_redis_client()
    client.lpush(REDIS_KEY_PREDICTIONS, json.dumps(entry, ensure_ascii=False))
    client.expire(REDIS_KEY_PREDICTIONS, REDIS_TTL_SECONDS)


def load_predictions_from_redis(limit: int = 50) -> List[Dict[str, Any]]:
    client = get_redis_client()
    raw_items = client.lrange(REDIS_KEY_PREDICTIONS, 0, limit - 1)
    results: List[Dict[str, Any]] = []
    for raw in raw_items:
        try:
            results.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return results


def main():
    st.title("Dashboard del modelo de cáncer de tórax")

    tab_metrics, tab_inference, tab_history = st.tabs(
        ["Métricas y entrenamiento", "Análisis de imagen con IA", "Histórico de análisis"]
    )

    # =========================
    # 1) MÉTRICAS Y ENTRENAMIENTO
    # =========================
    with tab_metrics:
        st.header("Resumen de modelos")

        col_eff, col_mob = st.columns(2)

        eff_info = None
        mob_info = None

        with col_eff:
            st.subheader("EfficientNet (modelo base)")
            try:
                eff_info = load_efficient_info()
            except FileNotFoundError:
                st.error(
                    "No se encontró `samples/metrics.json`. "
                    "Entrena y evalúa el modelo EfficientNet primero."
                )
            if eff_info is not None:
                eff_metrics = eff_info.get("metrics", {})
                eff_acc = eff_metrics.get("accuracy")
                eff_prec = eff_metrics.get("precision")
                eff_rec = eff_metrics.get("recall")
                eff_f1 = eff_metrics.get("f1_score")

                m1, m2, m3, m4 = st.columns(4)
                if eff_acc is not None:
                    m1.metric("Accuracy", f"{eff_acc*100:.2f}%")
                if eff_prec is not None:
                    m2.metric("Precisión", f"{eff_prec*100:.2f}%")
                if eff_rec is not None:
                    m3.metric("Recall", f"{eff_rec*100:.2f}%")
                if eff_f1 is not None:
                    m4.metric("F1-Score", f"{eff_f1*100:.2f}%")

                st.caption("Clases")
                st.write(eff_info.get("class_names", []))

                history_path = HISTORY_PATH
                if isinstance(history_path, Path):
                    history_exists = history_path.exists()
                else:
                    history_exists = Path(history_path).exists()
                if history_exists:
                    st.image(
                        str(history_path),
                        caption="Historial EfficientNet",
                        use_column_width=True,
                    )

        with col_mob:
            st.subheader("MobileViT (modelo ligero)")
            try:
                mob_info = load_mobilevit_info()
            except FileNotFoundError:
                st.error(
                    "No se encontró `samples/metrics_mobilevit.json`. "
                    "Entrena y evalúa el modelo MobileViT primero (`python main2.py`)."
                )
            if mob_info is not None:
                mob_metrics = mob_info.get("metrics", {})
                mob_acc = mob_metrics.get("accuracy")
                mob_prec = mob_metrics.get("precision")
                mob_rec = mob_metrics.get("recall")
                mob_f1 = mob_metrics.get("f1_score")

                m1, m2, m3, m4 = st.columns(4)
                if mob_acc is not None:
                    m1.metric("Accuracy", f"{mob_acc*100:.2f}%")
                if mob_prec is not None:
                    m2.metric("Precisión", f"{(mob_prec or 0)*100:.2f}%")
                if mob_rec is not None:
                    m3.metric("Recall", f"{(mob_rec or 0)*100:.2f}%")
                if mob_f1 is not None:
                    m4.metric("F1-Score", f"{(mob_f1 or 0)*100:.2f}%")

                st.caption("Clases")
                st.write(mob_info.get("class_names", []))

                if MOBILEVIT_HISTORY_PATH.exists():
                    st.image(
                        str(MOBILEVIT_HISTORY_PATH),
                        caption="Historial MobileViT",
                        use_column_width=True,
                    )

        # Comparación rápida por accuracy
        if eff_info is not None and mob_info is not None:
            eff_acc = eff_info["metrics"].get("accuracy", 0)
            mob_acc = mob_info["metrics"].get("accuracy", 0)
            mejor = "EfficientNet" if eff_acc >= mob_acc else "MobileViT"
            st.markdown(
                f"**Modelo con mejor accuracy global:** {mejor} "
                f"(EfficientNet: {eff_acc*100:.2f}%, MobileViT: {mob_acc*100:.2f}%)"
            )

    # =========================
    # 2) ANÁLISIS DE IMAGEN
    # =========================
    with tab_inference:
        st.header("Análisis de imagen con ambos modelos")
        st.write(
            "Sube una imagen de tórax para que **ambos modelos** (EfficientNet y MobileViT) "
            "la analicen y comparen sus predicciones."
        )

        uploaded_file = st.file_uploader(
            "Selecciona una imagen (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            # Guardar imagen en carpeta public con nombre único
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            ext = uploaded_file.name.split(".")[-1].lower()
            filename = f"thorax_{timestamp}.{ext}"
            save_path = PUBLIC_DIR / filename
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Cargar info de ambos modelos (se asume mismas clases)
            try:
                eff_info = load_efficient_info()
                class_names = eff_info.get("class_names", [])
                img_size = eff_info.get("img_size", IMG_SIZE)
            except FileNotFoundError:
                st.error(
                    "No se encontró `samples/metrics.json`. No es posible realizar "
                    "la inferencia sin conocer las clases del modelo EfficientNet."
                )
                class_names = []
                img_size = IMG_SIZE

            # Mostrar imagen original
            st.subheader("Imagen de entrada")
            image_placeholder = st.empty()

            # Preprocesar imagen
            img_array, pil_image = preprocess_image(save_path, img_size)
            image_placeholder.image(
                pil_image,
                caption=f"Imagen cargada ({filename})",
                use_column_width=False,
            )

            # Cargar modelos
            try:
                eff_model = load_efficient_model()
            except OSError:
                st.error(
                    "No se encontró el modelo EfficientNet. "
                    "Asegúrate de haber entrenado y guardado `model/model_efficient.keras`."
                )
                return

            try:
                mob_model = load_mobilevit_model()
            except OSError:
                st.error(
                    "No se encontró el modelo MobileViT. "
                    "Asegúrate de haber entrenado y guardado `model/model_mobilevit.keras`."
                )
                return

            if not class_names:
                st.warning(
                    "No se encontraron nombres de clases en `metrics.json`. "
                    "No se pueden mostrar resultados interpretables."
                )
                return

            # Inferencia con ambos modelos
            with st.spinner("Realizando predicciones con ambos modelos..."):
                eff_preds = eff_model.predict(img_array, verbose=0)[0]
                mob_preds = mob_model.predict(img_array, verbose=0)[0]

            # Resultados lado a lado
            col_e, col_m = st.columns(2)

            eff_idx = int(np.argmax(eff_preds))
            eff_class = class_names[eff_idx]
            eff_conf = float(eff_preds[eff_idx])

            mob_idx = int(np.argmax(mob_preds))
            mob_class = class_names[mob_idx]
            mob_conf = float(mob_preds[mob_idx])

            with col_e:
                st.subheader("EfficientNet")
                st.write(f"**Clase predicha:** {eff_class}")
                st.write(f"**Confianza:** {eff_conf*100:.2f}%")

                df_eff = pd.DataFrame(
                    {"Clase": class_names, "Probabilidad": eff_preds}
                )
                df_eff["Probabilidad (%)"] = df_eff["Probabilidad"] * 100
                st.dataframe(
                    df_eff[["Clase", "Probabilidad (%)"]].sort_values(
                        "Probabilidad (%)", ascending=False
                    ),
                    use_container_width=True,
                )
                st.bar_chart(df_eff.set_index("Clase")["Probabilidad"])

            with col_m:
                st.subheader("MobileViT")
                st.write(f"**Clase predicha:** {mob_class}")
                st.write(f"**Confianza:** {mob_conf*100:.2f}%")

                df_mob = pd.DataFrame(
                    {"Clase": class_names, "Probabilidad": mob_preds}
                )
                df_mob["Probabilidad (%)"] = df_mob["Probabilidad"] * 100
                st.dataframe(
                    df_mob[["Clase", "Probabilidad (%)"]].sort_values(
                        "Probabilidad (%)", ascending=False
                    ),
                    use_container_width=True,
                )
                st.bar_chart(df_mob.set_index("Clase")["Probabilidad"])

            # Modelo más seguro para ESTA imagen
            if eff_conf > mob_conf:
                best_model_instance = "EfficientNet"
            elif mob_conf > eff_conf:
                best_model_instance = "MobileViT"
            else:
                best_model_instance = "Empate (misma confianza)"

            st.markdown(
                f"**Modelo más seguro para esta imagen:** {best_model_instance} "
                f"(EfficientNet: {eff_conf*100:.2f}%, MobileViT: {mob_conf*100:.2f}%)"
            )

            # Guardar en Redis
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "image_filename": filename,
                "image_path": str(save_path),
                "efficientnet": {
                    "predicted_class": eff_class,
                    "confidence": eff_conf,
                    "predictions": {
                        class_names[i]: float(eff_preds[i])
                        for i in range(len(class_names))
                    },
                },
                "mobilevit": {
                    "predicted_class": mob_class,
                    "confidence": mob_conf,
                    "predictions": {
                        class_names[i]: float(mob_preds[i])
                        for i in range(len(class_names))
                    },
                },
                "best_model_for_image": best_model_instance,
            }
            save_prediction_to_redis(entry)

    # =========================
    # 3) HISTÓRICO DE ANÁLISIS (Redis)
    # =========================
    with tab_history:
        st.header("Histórico de análisis almacenado")

        items = load_predictions_from_redis(limit=100)
        if not items:
            st.info(
                "Aún no hay registros en Redis o han expirado. "
                "Realiza un análisis de imagen en la pestaña anterior."
            )
        else:
            for item in items:
                with st.expander(
                    f"{item.get('timestamp', '')} - {item.get('image_filename', '')}"
                ):
                    img_path = item.get("image_path")
                    if img_path and Path(img_path).exists():
                        st.image(
                            img_path,
                            caption=item.get("image_filename", ""),
                            use_column_width=False,
                        )

                    st.write(f"**Modelo más seguro:** {item.get('best_model_for_image')}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**EfficientNet**")
                        st.write(
                            f"Clase: {item['efficientnet']['predicted_class']}, "
                            f"Confianza: {item['efficientnet']['confidence']*100:.2f}%"
                        )
                    with col2:
                        st.markdown("**MobileViT**")
                        st.write(
                            f"Clase: {item['mobilevit']['predicted_class']}, "
                            f"Confianza: {item['mobilevit']['confidence']*100:.2f}%"
                        )


if __name__ == "__main__":
    main()

