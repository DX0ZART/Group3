# Sentiment Analysis with Deep Learning

QMSS5074GR - Project 3

## Team Information
- **Group ID**: Group 3
- **UNIs**: xl3456, yz4661, lz3073
- **Names**: Xiaole Liu, Evan Zhou, Leizu Zhang

## Project Overview

Sentiment analysis on the Stanford Sentiment Treebank (SST-2) dataset, comparing traditional ML, deep learning, and transfer learning approaches.

## Implementation

### Part 1: Data Preprocessing
- Text cleaning (HTML removal, lowercasing, punctuation)
- TF-IDF vectorization
- Sequence tokenization and padding
- Saved vectorizers for inference

### Part 2: Exploratory Data Analysis
- Class distribution analysis
- Text length statistics
- Word frequency and word clouds
- Correlation analysis

### Part 3: Traditional Models
- Logistic Regression (hyperparameter tuned)
- Linear SVM (hyperparameter tuned)
- Random Forest
- Gradient Boosting

### Part 4: Neural Networks
- MLP (trainable/frozen embeddings)
- 1D-CNN
- LSTM
- Bi-LSTM

### Part 5: Transfer Learning
- GloVe pre-trained embeddings
- BERT fine-tuning

### Part 6: Hyperparameter Optimization
- Keras Tuner with random search
- 6 hyperparameters optimized

### Part 7: Analysis
- Model comparison
- McNemar's statistical test
- Error analysis with examples

### Part 8: Bonus Extensions
- Data augmentation
- VADER sentiment features
- Flask API deployment

## Quick Start

### Installation
```bash
pip install datasets scikit-learn pandas numpy matplotlib seaborn wordcloud
pip install tensorflow transformers keras-tuner
pip install xgboost joblib nlpaug vaderSentiment flask
```

### Run
```bash
jupyter notebook Group3_project3.ipynb
```

## API Deployment

```bash
python sentiment_api.py

curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This movie was amazing!"}'
```
