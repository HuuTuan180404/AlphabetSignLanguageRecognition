# Alphabet Sign Language Recognition

This project implements a machine learning system for recognizing alphabet signs in sign language using computer vision and deep learning techniques.

## Project Overview

This system can recognize hand gestures representing letters of the alphabet in sign language from images. The project uses machine learning models to classify different hand gestures into their corresponding alphabet letters.

## Project Structure

```
├── data/
│   ├── dataset/         # Contains image data for each alphabet sign
│   ├── create_data.ipynb    # Notebook for data preparation
│   ├── DATASET.pickle   # Processed dataset
│   ├── utils.py         # Utility functions for data handling
│   └── main.ipynb       # Main data processing notebook
├── model/
│   ├── attention.py     # Attention mechanism implementation
│   └── model.py         # Model architecture definition
├── out-checkpoints/     # Saved model checkpoints
├── environment.yml      # Conda environment configuration
├── main.ipynb          # Main project notebook
├── train.py           # Training script
├── train_libs.py      # Training utilities
└── test_libs.py       # Testing utilities
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/HuuTuan180404/AlphabetSignLanguageRecognition.git
cd AlphabetSignLanguageRecognition
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate <environment_name>
```

## Usage

1. **Data Preparation**: 
   - Use `data/create_data.ipynb` to prepare your dataset
   - Images should be organized in the `data/dataset/` directory by alphabet letter

2. **Training**:
   - Run training using:
   ```bash
   python train.py
   ```
   - Model checkpoints will be saved in `out-checkpoints/`

3. **Testing and Real-time Recognition**:
   - Activate the conda environment:
   ```bash
   conda activate alphabetic
   ```
   - Run real-time testing with webcam:
   ```bash
   python test_libs.py
   ```
   - The program will:
     - Open your webcam
     - Use MediaPipe to detect hand gestures
     - Recognize and display the corresponding alphabet letter in real-time
     - Press 'q' to quit the program
   
   Requirements for testing:
   - Working webcam
   - Trained model in `out-checkpoints/` directory
   - Good lighting conditions for better recognition

   For offline testing and evaluation:
   - Use `main.ipynb` for batch testing and inference
   - Models can be loaded from saved checkpoints

## Dataset

The dataset consists of images of hand gestures representing alphabet letters. Each letter has its own directory containing multiple images showing the corresponding hand sign. The dataset is organized in the `data/dataset/` directory, with separate folders for each alphabet letter (A-Z).

## Model Architecture

The project implements a deep learning model with attention mechanism for improved recognition accuracy. The model architecture is defined in `model/model.py` and uses an attention mechanism implemented in `model/attention.py` for better feature extraction and recognition performance.

## Results

Model checkpoints and training results can be found in the `out-checkpoints/` directory. The system is designed to recognize alphabet signs accurately, helping bridge communication gaps for sign language users.

## Contributors

- HuuTuan180404

## License

[MIT License]
