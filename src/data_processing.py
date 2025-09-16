import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

class DataProcessor:
    def __init__(self):
        pass
   
    def visualize_random_image(self, data_path):
        """Display a random image from the dataset """
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
       
        species = random.choice(os.listdir(data_path))
        img_choice = random.choice(os.listdir(os.path.join(data_path, species)))
        img_path = os.path.join(data_path, species, img_choice)
       
        img = mimg.imread(img_path)
        plt.title(species)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
       
        return species, img_path
   
    def create_data_generators(self, data_path, img_size=(224, 224), batch_size=32):
        """Create data generators"""
        train_path = os.path.join(data_path, 'train')
        valid_path = os.path.join(data_path, 'valid')
        test_path = os.path.join(data_path, 'test')
       
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
       
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
       
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
       
        validation_generator = test_datagen.flow_from_directory(
            valid_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
       
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
       
        return train_generator, validation_generator, test_generator
   
    def get_class_names(self, generator):
        """Get class names from generator"""
        return list(generator.class_indices.keys())
   
    def save_class_names(self, class_names, file_path):
        """Save class names to file"""
        try:
        
            dir_name = os.path.dirname(file_path)
           
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"Created directory: {dir_name}")
           
          
            with open(file_path, 'w', encoding='utf-8') as f:
                for class_name in class_names:
                    f.write(f"{class_name}\n")
            print(f"Class names saved to: {file_path}")
            return True
        except Exception as e:
            print(f"Error saving class names to {file_path}: {e}")
           
          
            try:
                alt_path = 'class_names.txt'
                with open(alt_path, 'w', encoding='utf-8') as f:
                    for class_name in class_names:
                        f.write(f"{class_name}\n")
                print(f"Class names saved to alternative path: {alt_path}")
                return True
            except Exception as alt_e:
                print(f"Error saving to alternative path: {alt_e}")
                return False
   
    def load_class_names(self, file_path):
        """Load class names from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines()]
            return class_names
        except Exception as e:
            print(f"Error loading class names from {file_path}: {e}")
            return None