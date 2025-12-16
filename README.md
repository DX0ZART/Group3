# ML Project 3: Sentiment Analysis with Deep Learning

## Project Overview

This project implements a comprehensive sentiment analysis system using the Stanford Sentiment Treebank (SST-2) dataset, comparing traditional machine learning, deep learning, and transfer learning approaches.

## Files

- **`project3.ipynb`** - Main Jupyter notebook with all implementations
- **`MODEL_MANAGEMENT.md`** - Guide for managing saved models
- **`tfidf_vectorizer.joblib`** - Saved TF-IDF vectorizer
- **`neural_tokenizer.joblib`** - Saved Keras tokenizer
- **`models/`** - Directory containing all trained models
- **`sentiment_api.py`** - Flask API for model deployment (generated)

## Features

### ✅ Automatic Model Caching
- **No Retraining**: Models are saved after training and automatically loaded on subsequent runs
- **Time Saving**: Rerunning the notebook takes minutes instead of hours
- **Smart Detection**: Each training cell checks for existing models before training

### 📊 Complete Implementation

**Part 1: Data Preprocessing**
- SST-2 dataset from GLUE benchmark
- Stratified train/val/test splits
- Text cleaning pipeline
- TF-IDF and sequence vectorization

**Part 2: EDA**
- Class distribution analysis
- Review length statistics
- Word frequency and word clouds
- Correlation analysis

**Part 3: Traditional Models**
- Logistic Regression (with hyperparameter tuning)
- Linear SVM (with hyperparameter tuning)
- Random Forest
- Gradient Boosting
- Feature importance analysis

**Part 4: Neural Networks**
- MLP with trainable/frozen embeddings
- 1D-CNN for text classification
- LSTM and Bi-LSTM models
- Performance comparisons

**Part 5: Transfer Learning**
- GloVe pre-trained embeddings
- BERT fine-tuning
- Comparison with from-scratch models

**Part 6: Hyperparameter Optimization**
- Keras Tuner integration
- Random search over 6 hyperparameters
- Visualization of tuning results

**Part 7: Analysis**
- Comprehensive model comparison
- McNemar's statistical significance test
- Error analysis with 20+ examples
- Detailed discussion of 5 misclassifications

**Part 8: Bonus Extensions**
- Data augmentation (synonym swapping)
- VADER sentiment lexicon features
- Flask REST API deployment

## Quick Start

### Installation

```bash
# Install required packages
pip install datasets scikit-learn pandas numpy matplotlib seaborn wordcloud
pip install tensorflow transformers keras-tuner optuna
pip install xgboost joblib nlpaug vaderSentiment flask
```

### Running the Notebook

1. **First Time (Full Training)**
   ```bash
   jupyter notebook project3.ipynb
   ```
   - Run all cells sequentially
   - This will train all models and save them
   - Takes 2-4 hours depending on hardware

2. **Subsequent Runs (Fast)**
   - Re-run any cell - it will load saved models
   - Only cells with code changes will retrain
   - Takes 5-10 minutes

### Model Management

```python
# List all saved models
list_saved_models()

# Delete a specific model to retrain
clear_specific_model('mlp_trainable.h5')

# Delete all models (use with caution!)
clear_all_models()
```

See [`MODEL_MANAGEMENT.md`](MODEL_MANAGEMENT.md) for detailed guide.

## Results Summary

### Best Models (Example - run notebook for actual results)

| Category | Model | F1-Score | Accuracy | ROC-AUC |
|----------|-------|----------|----------|---------|
| Overall | BERT | ~0.92 | ~0.92 | ~0.97 |
| Traditional | Linear SVM | ~0.83 | ~0.82 | ~0.89 |
| Neural | CNN (Tuned) | ~0.90 | ~0.89 | ~0.95 |

*Note: Actual values will vary based on training runs*

### Key Findings

1. **Transfer Learning Wins**: BERT and GloVe significantly outperform from-scratch models
2. **Linear Models Strong**: SVM and Logistic Regression surprisingly competitive
3. **CNNs Beat RNNs**: 1D-CNNs faster and often more accurate than LSTMs
4. **Tuning Matters**: Hyperparameter optimization improved CNN by 2-3%
5. **Length ≠ Sentiment**: Review length has minimal correlation with sentiment

## Project Structure

```
Project_3/
├── project3.ipynb              # Main notebook
├── MODEL_MANAGEMENT.md         # Model management guide
├── README.md                   # This file
├── requirements.txt            # Project requirements
├── models/                     # Saved models (auto-created)
│   ├── logistic_regression.joblib
│   ├── linear_svm.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   ├── mlp_trainable.h5
│   ├── mlp_frozen.h5
│   ├── cnn_model.h5
│   ├── lstm_model.h5
│   ├── bilstm_model.h5
│   ├── cnn_glove_frozen.h5
│   ├── cnn_glove_trainable.h5
│   ├── best_tuned_model.h5
│   └── bert_model/
├── keras_tuner/                # Tuning results (auto-created)
├── tfidf_vectorizer.joblib     # Saved vectorizer
├── neural_tokenizer.joblib     # Saved tokenizer
└── sentiment_api.py            # Flask API (auto-generated)
```

## Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 8 GB
- Disk: 5 GB free space
- Time: 3-4 hours (first run)

### Recommended
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA with 4+ GB VRAM (for BERT)
- Disk: 10 GB free space
- Time: 1-2 hours (first run)

## Customization

### To Retrain Specific Models

1. Delete the model file:
   ```python
   clear_specific_model('cnn_model.h5')
   ```

2. Re-run the corresponding training cell

### To Change Hyperparameters

1. Edit the hyperparameter values in the code
2. Delete the saved model
3. Re-run the training cell

### To Use Different Data

1. Modify the data loading cell (Part 1.1)
2. Delete all saved models and vectorizers
3. Re-run the entire notebook

## API Deployment

After running the notebook, deploy the model:

```bash
# Start the Flask API
python sentiment_api.py

# Test the API
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This movie was amazing!"}'
```

Response:
```json
{
  "text": "This movie was amazing!",
  "sentiment": "Positive",
  "confidence": 0.9234
}
```

## Common Issues

### Out of Memory
- Reduce batch size in training cells
- Close other applications
- Train one model at a time

### CUDA/GPU Errors
- Install compatible TensorFlow-GPU version
- Check CUDA/cuDNN installation
- Fallback to CPU by setting: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

### GloVe Download Fails
- Download manually from: https://nlp.stanford.edu/data/glove.6B.zip
- Extract `glove.6B.100d.txt` to project directory

### BERT Training Slow
- Reduce epochs (3 → 2)
- Reduce batch size (16 → 8)
- Use a smaller model (e.g., DistilBERT)

## Tips for Success

1. **Run Once Fully**: Let the notebook run completely at least once
2. **Monitor Progress**: Watch for "✓ Model saved" messages
3. **Check Models**: Use `list_saved_models()` to verify
4. **GPU for BERT**: Use GPU if available for BERT training
5. **Save Often**: The notebook auto-saves, but commit important changes

## Submission Checklist

- [ ] All cells executed successfully
- [ ] All models trained and evaluated
- [ ] Error analysis completed
- [ ] Visualizations generated
- [ ] README updated with results
- [ ] Code commented and clean
- [ ] Notebook exported to PDF
- [ ] GitHub repo created and shared
- [ ] Group info filled in notebook header

## Team Information

Update in the notebook:
- **Group ID**: [Fill Here]
- **UNIs**: [Fill Here]
- **Names**: [Fill Here]
- **GitHub**: [Fill Here]

## License

This project is for educational purposes as part of QMSS5074GR.

## Contact

For questions about model management or implementation details, refer to the comments in the notebook cells.

---

**Last Updated**: December 2024
**Course**: QMSS5074GR - Machine Learning
**Assignment**: Final Project (3rd)
# Group3
