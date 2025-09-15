import os
import sys
import tkinter as tk
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.prediction import Predictor
from src.gui import BirdSpeciesGUI

def main():
    # Initialize processors
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()
    predictor = Predictor()
   
    # Initialize accuracy and time variables
    model_accuracy = None
    training_time = None
    best_accuracy = None
   
   
    base_path = r'E:\birds_classificationAI\data'
    models_dir = 'models'
    model_path = os.path.join(models_dir, 'bird_species_model.h5')
    best_model_path = os.path.join(models_dir, 'best_model.h5')
    class_names_path = os.path.join(models_dir, 'class_names.txt')
    training_info_path = os.path.join(models_dir, 'training_info.txt')
   
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
   
    # Check if trained model exists
    if os.path.exists(best_model_path) and os.path.exists(class_names_path):
        print("Found pre-trained model. Loading the best model...")
        model = model_trainer.load_model(best_model_path)
        class_names = data_processor.load_class_names(class_names_path)
       
        # Try to load accuracy and time information if saved
        try:
            if os.path.exists(training_info_path):
                with open(training_info_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 3:
                        training_time = float(lines[0].strip())
                        model_accuracy = float(lines[1].strip())
                        best_accuracy = float(lines[2].strip())
                        print(f"Loaded training info: Time={training_time}, Accuracy={model_accuracy}, Best Accuracy={best_accuracy}")
        except Exception as e:
            print(f"Error loading training info: {e}")
           
    else:
        print("No trained model found. Starting training...")
       
        # Process data
        train_generator, validation_generator, test_generator = data_processor.create_data_generators(base_path)
        class_names = data_processor.get_class_names(train_generator)
       
        # Save class names
        data_processor.save_class_names(class_names, class_names_path)
       
        # Train model
        model, history, training_time, best_accuracy = model_trainer.train_model(
            train_generator, validation_generator, len(class_names)
        )
       
        # Save the final model
        model_trainer.save_model(model, model_path)
       
        # Evaluate model
        test_loss, test_accuracy = model_evaluator.evaluate_model(model, test_generator)
        model_accuracy = test_accuracy
       
        # Save training information
        try:
            with open(training_info_path, 'w') as f:
                f.write(f"{training_time}\n")
                f.write(f"{model_accuracy}\n")
                f.write(f"{best_accuracy}\n")
            print(f"Training info saved to: {training_info_path}")
        except Exception as e:
            print(f"Error saving training info: {e}")
       
        # Print results
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Final model accuracy on test data: {test_accuracy*100:.2f}%")
        print(f"Best validation accuracy during training: {best_accuracy*100:.2f}%")
   
    # Initialize predictor with model and class names
    predictor.initialize(model, class_names)
   
    # Launch GUI with accuracy and time information
    root = tk.Tk()
    app = BirdSpeciesGUI(root, predictor, model_accuracy, training_time, best_accuracy)
    root.mainloop()

if __name__ == "__main__":
    main()