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
| 202274261    | [Saleh ahmed alhowat]    | Data analysis        | Data processing and preparing    |
| 202174374    | [waleed yahya al_sharafi]| ML Engineer          |  Model architectureand training  |
| 202073349    | [taha ahmed alqutami]    | Frontend Developer   |GUI implementation and data visulization |



Prerequisites

· Python 3.9.2 
· UV package manager

Installation Steps

1. Clone the repository:

git clone <https://github.com/saleh202274261/Bird-Species-Classification.git>
cd bird-species-classification

1. Install dependencies using UV:

uv sync


1. Run the project:


uv run python main.py


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

· Model Accuracy: XX%
· Training Time: XX minutes
· Key Findings: [Brief summary of results]

