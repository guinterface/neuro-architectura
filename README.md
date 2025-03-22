🧠 Neuro Architectura

This repository contains scripts and documentation for a project focused on applying machine learning regression techniques to EEG data. The goal is to predict continuous target variables (e.g., cognitive scores, stimuli strength) based on preprocessed EEG signals.
📁 Project Structure

eeg-regression/
│
├── data/                    # Folder to store raw or processed EEG data (not included in repo)
├── preprocessing/           # Scripts for data cleaning, filtering, and feature extraction
│   └── preprocess.py
│
├── models/                  # Machine learning model definitions and training scripts
│   └── train_model.py
│
├── utils/                   # Helper functions for loading data, metrics, etc.
│   └── metrics.py
│
├── results/                 # Folder for saving model outputs, plots, and evaluations
├── requirements.txt         # Python dependencies
└── README.md                # You're here!

⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/eeg-regression.git
cd eeg-regression
pip install -r requirements.txt

🧪 Usage
1. Preprocess the EEG Data

python preprocessing/preprocess.py

2. Train the Model

python models/train_model.py

📊 Results

Results such as loss curves and model performance are saved in the results/ folder.
📚 Dataset

    Note: Due to size/privacy, the dataset is not included here.

You can download the dataset and place it in the data/ folder. See Project Report for details.
🔗 Citation

If you use this code or reference this project in your research, please cite:

Guilherme Simioni Bonfim. (2025). Neuro Architectura. GitHub repository: https://github.com/guinterface/neuro-architectura

📬 Contact

For questions, feel free to open an issue or reach out to gsbonfim@stanford.edu
