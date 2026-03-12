import tensorflow as tf
from keras import Input, Model
from keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    Rescaling,
    Reshape,
)


def _mobilevit_block(x, dim: int, num_heads: int = 4, mlp_ratio: int = 4):
    """
    Bloque simplificado tipo MobileViT:
    - Convoluciones locales
    - Atención tipo Transformer sobre parches
    """
    # Proyección local
    x = Conv2D(dim, kernel_size=3, padding="same", activation="relu")(x)

    # Forma [B, H, W, C] -> [B, N, C] usando sólo capas Keras
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[-1]

    if h is None or w is None or c is None:
        raise ValueError(
            "Las dimensiones espaciales deben ser conocidas para el bloque MobileViT."
        )

    n = int(h) * int(w)
    seq = Reshape((n, int(c)))(x)

    # Auto-atención
    y = LayerNormalization()(seq)
    y = MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)(y, y)
    seq = Add()([seq, y])

    # MLP
    y = LayerNormalization()(seq)
    y = Dense(mlp_ratio * dim, activation="relu")(y)
    y = Dense(dim)(y)
    seq = Add()([seq, y])

    # Volvemos a [B, H, W, C]
    x = Reshape((int(h), int(w), dim))(seq)

    # Fusión a canales de entrada
    x = Conv2D(dim, kernel_size=1, padding="same", activation="relu")(x)

    return x


def create_model_mobilevit(num_classes: int, img_size: int) -> Model:
    """
    Modelo compacto inspirado en MobileViT para clasificación de imágenes.
    Mantiene un número moderado de parámetros para ser ligero.
    """
    inputs = Input(shape=(img_size, img_size, 3))

    # Normalización a [0,1]
    x = Rescaling(1.0 / 255.0)(inputs)

    # Stem convolucional con reducción progresiva de resolución
    # 224x224 -> 112x112
    x = Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = Conv2D(48, 1, activation="relu")(x)

    # 112x112 -> 56x56 (solo convoluciones, sin atención para evitar uso masivo de memoria)
    x = DepthwiseConv2D(3, strides=2, padding="same", activation="relu")(x)
    x = Conv2D(64, 1, activation="relu")(x)

    # 56x56 -> 28x28
    x = DepthwiseConv2D(3, strides=2, padding="same", activation="relu")(x)
    x = Conv2D(96, 1, activation="relu")(x)

    # Único bloque MobileViT a baja resolución (28x28 -> 784 tokens)
    x = _mobilevit_block(x, dim=96, num_heads=2, mlp_ratio=2)

    # Cabezal de clasificación
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="mobilevit_classifier")
    return model

