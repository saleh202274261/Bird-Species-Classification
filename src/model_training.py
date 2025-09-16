import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time
import os

class ModelTrainer:
    def __init__(self):
        pass
   
    def build_model(self, num_classes, input_shape=(224, 224, 3)):
        """Build model (exactly matching original code)"""
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None
        )
       
        base_model.trainable = False
       
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.2),
            Dense(2560, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1280, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
       
        return model
   
    def compile_model(self, model, learning_rate=0.001):
        """Compile model (exactly matching original code)"""
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
       
        return model
   
    def train_model(self, train_generator, validation_generator, num_classes, epochs=50):
        """Train model with best model saving"""
        model = self.build_model(num_classes)
        model = self.compile_model(model)
       
        # Define callbacks with model checkpoint to save the best model
        best_model_path = os.path.join('models', 'best_model.h5')
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-7),
            ModelCheckpoint(
                best_model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
       
        # Start training timer
        start_time = time.time()
       
        # Train the model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
       
        # Calculate training time
        training_time = time.time() - start_time
       
        # Get the best validation accuracy
        best_accuracy = max(history.history['val_accuracy'])
       
        # Load the best model saved by ModelCheckpoint
        best_model = load_model(best_model_path)
       
        return best_model, history, training_time, best_accuracy
   
    def save_model(self, model, model_path):
        """Save model"""
        try:
           
            dir_name = os.path.dirname(model_path)
           
        
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"Created directory: {dir_name}")
           
       
            model.save(model_path)
            print(f"Model saved to: {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model to {model_path}: {e}")
           
        
            try:
                alt_path = 'bird_species_model.h5'
                model.save(alt_path)
                print(f"Model saved to alternative path: {alt_path}")
                return True
            except Exception as alt_e:
                print(f"Error saving to alternative path: {alt_e}")
                return False
   
    def load_model(self, model_path):
        """Load model"""
        try:
            return load_model(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None