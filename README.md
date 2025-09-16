Bird Species Classification

Project Description

This project is a deep learning application that classifies bird species from images using a convolutional neural network (CNN) based on EfficientNetB0 with transfer learning. The application includes a user-friendly GUI that allows users to upload bird images and receive accurate species predictions with confidence scores.

Key Features:

· Image preprocessing and augmentation
· Transfer learning with EfficientNetB0
· Model training with early stopping and checkpointing
· Interactive GUI for predictions
· Performance metrics visualization

Team Information

| AC.NO        | Name                     | Role                 | Contributions |
|--------------|--------------------------|----------------------|---------------|
| 202274261    |   Saleh ahmed alhowat    | Data analysis        | Data processing and preparing    |
| 202174374    |   waleed yahya al_sharafi| ML Engineer          |  Model architectureand training  |
| 202073349    |   taha ahmed alqutami    | Frontend Developer   |GUI implementation and data visulization |



Prerequisites

· Python 3.9.2 
· UV package manager

Installation Steps

1. Clone the repository:

git clone <https://github.com/saleh202274261/Bird-Species-Classification.git>
cd bird-species-classification

1. Install dependencies using UV:

uv sync

2. Download the dataset (CUB_200_2011) and decompress the zibfile and run the data_split.py in the same file that the date is in. this code will divided the data into three categories train/valid/test take this file put it in the data file in the project structure. 
3. Run the project:

1.uv run python main.py

2.Using the GUI
1.Run application : uv run python main.py
2.Click "chose image" to select a bird image
3.view the top prediction with confidence scores 
4.see model accuracy in the information pannel

Project Structure


```bird-species-classification/
├── README.md              
├── pyproject.toml         
├── .python-version        
├── .gitignore             
├── main.py               
├── src/                  
│   ├── data_processing.py 
│   ├── model_training.py  
│   ├── model_evaluation.py
│   ├── prediction.py      
│   └── gui.py            
├── models/                
│   ├── best_model.h5      
│   ├── class_names.txt    
│   └── training_info.txt  
├── notebooks/            
├── data/                
└── docs/ ```               


Usage

Basic Usage

```python
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer

# Load and process data
data_processor = DataProcessor()
train_generator, validation_generator, test_generator = data_processor.create_data_generators('data/')

# Train model
model_trainer = ModelTrainer()
model, history, training_time, best_accuracy = model_trainer.train_model(
    train_generator, validation_generator, num_classes=20
)
```


Running the Application


uv run python main.py


Results

· Model Accuracy: 71%
· Training Time: 5000 seconds
· Key Findings: 
- EfficientNetB0 provides excellent feature extraction for fine-grained bird classification
- Data augmentation significantly improves model generalization
- GlobalAveragePooling with custom dense layers outperforms traditional flattening approach
- The model handles class imbalance well despite varying species sample sizes

### Confusion Matrix
The model shows particular strength in distinguishing between visually distinct species, with some confusion occurring between similar-looking species within the same family.

