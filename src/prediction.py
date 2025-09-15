import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

class Predictor:
    def __init__(self):
        self.model = None
        self.class_names = None
   
    def initialize(self, model, class_names):
        """Initialize predictor with model and class names"""
        self.model = model
        self.class_names = class_names
   
    def predict_image(self, img_path):
        """Predict bird species in a given image"""
        if self.model is None or self.class_names is None:
            raise ValueError("Predictor not initialized. Please call initialize() first.")
       
        try:
            # Load and preprocess image
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
           
            # Predict
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
           
            # Get top 5 predictions
            top_5_indices = np.argsort(predictions[0])[-5:][::-1]
            top_5_confidences = predictions[0][top_5_indices]
            top_5_classes = [self.class_names[i] for i in top_5_indices]
           
            results = {
                'predicted_class': self.class_names[predicted_class],
                'confidence': float(confidence),
                'top_5': list(zip(top_5_classes, [float(conf) for conf in top_5_confidences]))
            }
           
            return results
        except Exception as e:
            raise Exception(f"Error predicting image: {str(e)}")