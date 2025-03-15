# Fake News Detection with RoBERTa
This project aims to evaluate the effectiveness of a RoBERTa-based model for detecting fake news by training and testing on three different datasets: ISOT, FakeNews-Kaggle, and LIAR. The goal is to determine which dataset leads to the best generalizability in fake news detection.

## Project Purpose**
The rise of misinformation has made automated fake news detection an important task. This project explores the effectiveness of using a RoBERTa model for binary classification (real vs. fake) by training separate models on three datasets. The results are evaluated using precision, recall, F1-score, accuracy, and explainability techniques like LIME analysis.

## Datasets Used**
The project utilizes the following datasets:

**ISOT**: A dataset containing fake and real news articles.
**FakeNews-Kaggle**: A dataset sourced from Kaggle with manually labeled news headlines.
**LIAR**: A dataset containing fact-checked political statements.

Each dataset was preprocessed and tokenized before training.

## Model Training
The RoBERTa model was fine-tuned separately on each dataset using PyTorch and Hugging Face's Transformers library.
- Tokenization was performed using RobertaTokenizer.
- Training was done on Talapas, using GPU resources.

## Training Instructions
To train the model, run the following command:

python train.py

This script loads the datasets, fine-tunes the model, and saves the trained weights.

## Results
Performance Metrics
The following metrics were used to compare models:
- Accuracy
- Precision
- Recall
- F1-score
- Weighted averages for balanced performance comparison
- Below is a summary of the model performance:

Dataset	            Precision	Recall	F1-Score	Accuracy
ISOT	                0.30	   0.50	    0.37	     0.60
FakeNews-Kaggle	      0.20	   0.50	    0.29	     0.40
LIAR	                0.30	   0.50	    0.37	     0.60

## Visualization of Model Performance
I visualized the classification results using:

**Bar Plots of Accuracy and F1-scores**

**Line Plots Comparing Precision, Recall, and F1-score**

**LIME Analysis for Model Explainability**

## LIME Analysis
A LIME (Local Interpretable Model-Agnostic Explanations) analysis was conducted on five sample titles to explain why the models classified certain news as real or fake. However, some models, particularly those trained on ISOT and FakeNews-Kaggle, exhibited biases towards classifying certain news as real or fake.

## Limitations
Dataset Bias: Some datasets had class imbalances, impacting model predictions.
Generalizability Issues: The model trained on one dataset did not always perform well on others.
Explainability Challenges: LIME visualizations revealed some inconsistencies in feature importance.
Hardware Constraints: Running evaluations on a CPU was time-consuming.

## Repository Structure
FakeNews-Detection/
│── models.py             # Code for defining and loading RoBERTa models

│── dataset.py            # Dataset processing class

│── train_models.py       # Training script for RoBERTa models

│── data/

│   ├── processed_ISOT.csv

│   ├── processed_FakeNews.csv

│   ├── processed_LIAR.csv

│── notebooks/

│   ├── data_demo.ipynb   # Demonstrates dataset processing

│   ├── evaluation.ipynb  # Evaluates models & visualizes results

│── README.md             # Project documentation

## How to Run the Evaluation
To evaluate the trained models, use:

jupyter notebook notebooks/evaluation.ipynb

This notebook loads trained models, makes predictions, and visualizes the results.


## Conclusion
This project demonstrates the challenges of fake news detection using different datasets. While RoBERTa showed promising results, dataset bias and domain-specificity issues impacted its generalizability. Future work could involve:
- Fine-tuning with more diverse datasets
- Using ensemble methods for better performance
- Exploring other transformer-based architectures like BERT and T5

