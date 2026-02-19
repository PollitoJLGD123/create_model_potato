import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import json
from src.constant.constants import MODEL_PATH, METRICS_PATH

class PotatoDiseaseClassifier:
    def __init__(self, model_path, info_path):
        # Cargar modelo
        self.model = tf.keras.models.load_model(model_path)
        
        # Cargar información
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        self.class_names = self.info['class_names']
        self.img_size = self.info['img_size']
    
    def predict(self, img_path):
        """
        Predice la enfermedad de una papa desde una imagen
        """
        # Cargar y preprocesar imagen
        img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocesar
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        # Predecir
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Resultados
        result = {
            'clase_predicha': self.class_names[predicted_class],
            'confianza': float(confidence),
            'todas_predicciones': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }
        
        return result
    
    def predict_batch(self, img_paths):
        """
        Predice múltiples imágenes
        """
        results = []
        for img_path in img_paths:
            result = self.predict(img_path)
            result['imagen'] = img_path
            results.append(result)
        return results

# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar clasificador
    classifier = PotatoDiseaseClassifier(MODEL_PATH, METRICS_PATH)
    
    # Predecir una imagen
    resultado = classifier.predict('image/healthy.JPG')
    print(f"Predicción: {resultado['clase_predicha']}")
    print(f"Confianza: {resultado['confianza']:.2%}")