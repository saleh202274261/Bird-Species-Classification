import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self):
        pass
   
    def evaluate_model(self, model, test_generator):
        """Evaluate model (matching original code)"""
        if model is None:
            print("Cannot evaluate: model is None")
            return None, None
           
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy:.4f}')
       
        y_pred = model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
       
        # Classification report
        class_names = list(test_generator.class_indices.keys())
        print('\nClassification Report:')
        print(classification_report(y_true, y_pred_classes, target_names=class_names))
       
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
       
        return test_loss, test_accuracy