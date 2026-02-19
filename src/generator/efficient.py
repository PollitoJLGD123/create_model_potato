import tensorflow as tf
from keras.applications import EfficientNetB0
from keras import Input, Model
from keras.applications.efficientnet import preprocess_input
from src.utils.general import create_data_augmentation
from keras.layers import GlobalAveragePooling2D, Dropout, Dense

def create_model_efficient(num_classes: int, img_size: int):
  
  base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(img_size, img_size, 3),
  )
  
  base_model.trainable = False
  
  # creamos el modelo completo
  inputs = Input(shape=(img_size, img_size, 3))
  
  ## aumento de datos
  data_augmentation = create_data_augmentation() ## aumentamos una capa para mejorar la generalizacion
  
  x = data_augmentation(inputs)
  
  x_normalized = preprocess_input(x) ## normalizamos las imagenes (si bien efficient ya lo hace, agregamos algo adicional)
  
  x_base_model = base_model(x_normalized, training=False) ## aplicamos el modelo base a las imagenes
  
  ## cabezal de clasificación (usar siempre x_base_model, no x)
  x_base_model = GlobalAveragePooling2D()(x_base_model)
  x_base_model = Dropout(0.2)(x_base_model)
  x_base_model = Dense(num_classes, activation='relu')(x_base_model)
  x_base_model = Dropout(0.3)(x_base_model)
  
  outputs = Dense(num_classes, activation='softmax')(x_base_model)
  
  model = Model(inputs, outputs)
  
  return model, base_model