from keras import Model
import tensorflow as tf
from src.constant.constants import LEARNING_RATE, MODEL_PATH, EPOCHS
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, CategoricalAccuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def train_model(model: Model, base_model: Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):
  
  model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall(), CategoricalAccuracy()],
  )
  
  ## callbacks
  callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint('best_model.keras', save_best_only=True), ## revisar
  ]
  
  print(f"Entrenando modelo {model.name}...")
  
  print(f"Empezamos con el cabezal del modelo base...")
  
  history1 = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)
  
  print(f"Segunda fase de entrenamiento...")
  
  base_model.trainable = True
  
  # Congelar las primeras capas (opcional)
  for layer in base_model.layers[:100]:
    layer.trainable = False
    
  model.compile(
      optimizer=Adam(learning_rate=LEARNING_RATE/10),
      loss='categorical_crossentropy',
      metrics=['accuracy', Precision(), Recall(), CategoricalAccuracy()],
  )
  
  history2 = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)
  
  # combinamos las historias
  
  history = {}
  
  for key in history1.history.keys():
    if key in history2.history:
      history[key] = history1.history[key] + history2.history[key]
  
  print(f"Entrenamiento completo...")
  
  return model, history
  
  