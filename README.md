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

| AC.NO        | Name                     | Role                        | Contributions |
|--------------|--------------------------|-----------------------------|---------------|
| 202274261    | [Saleh ahmed alhowat]    | Data analysis               | Data processing  |
| 202174374    | [waleed yahya al_sharafi]| ML Engineer                 |  Model architecture   
|                                                                           and training  |
| 202073349    | [taha ahmed alqutami]     | Frontend Developer         | GUI implementation, and |     |                                                                   data visulization |



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


bird-species-classification/
├── README.md              # Project documentation
├── pyproject.toml         # UV project configuration
├── .python-version          # Python version specification
├── .gitignore               # Git ignore rules
├── main.py               # Main application entry point
├── src/                  # Source code
│   ├── data_processing.py # Data processing modules
│   ├── model_training.py  # ML model implementations
│   ├── model_evaluation.py # Model evaluation
│   ├── prediction.py      # Prediction functions
│   └── gui.py            # GUI implementation
├── models/                  # Saved models directory
│   ├── best_model.h5        # Best trained model (created after training)
│   ├── class_names.txt      # Saved class names (created after training)
│   └── training_info.txt    # Training metrics (created after training)
├── notebooks/            # Jupyter notebooks
├── data/                # Dataset files
└── docs/                # Additional documentation


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


Running the Application


uv run python main.py


Results

· Model Accuracy: XX%
· Training Time: XX minutes
· Key Findings: [Brief summary of results]

Contributing

1. Fork the repository
2. Create a feature branch: git checkout -b feature-name
3. Make your changes
4. Commit changes: git commit -m 'Add feature'
5. Push to branch: git push origin feature-name
6. Submit a pull request