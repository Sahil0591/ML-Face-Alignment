# ML Face Alignment Project

This project implements a face alignment system using machine learning techniques in Python. It predicts facial landmarks from images, compares deep learning (using CNNs) and classical machine learning approaches (like Ridge regression), and visualizes results. The implementation is done using PyTorch and scikit-learn in a Jupyter Notebook.

## Prerequisites

- Python 3.8 or higher
- Jupyter Notebook

### Required Python Packages

Install all dependencies using:
```
pip install -r requirements.txt
```
Or, install manually:
```
pip install numpy torch torchvision scikit-learn matplotlib opencv-python
```
## Data

Place your training and test `.npz` files in a folder named `data`:
- data/face_alignment_training_images.npz
- data/face_alignment_test_images.npz

## How to Run

1. Activate your virtual environment (if using one):
   ``` 
   python -m venv venv
   ```
   ``` 
   venv\Scripts\activate  # On Windows
   ```
   or 
   ``` 
   source venv/bin/activate  # On macOS/Linux  
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```
4. Open the notebook (`face_alignment.ipynb`) on Jupyter and run all cells.

## Output

The notebook will generate a CSV file with predictions (`results_task2.csv`) in the `results` folder.

## Project Structure
```
.
├── data/
│   ├── face_alignment_training_images.npz
│   └── face_alignment_test_images.npz
├── results/
│   └── results_task2.csv
├── face_alignment.ipynb
├── requirements.txt
└── README.md
```

## Notes

- Do **not** upload the `venv` folder or `.ipynb_checkpoints` to GitHub.
- For any issues, please open an issue in this repository.
