# Sentiment Analysis 

## Project Overview
This Assignment performs sentiment analysis on Twitter data using a Bidirectional LSTM (BiLSTM) model. The dataset contains tweets labeled with different sentiment categories, and the model classifies tweets into four sentiment classes.

## Dataset
- **Source**: [Twitter Entity Sentiment Analysis dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **Columns**: `id`, `country`, `Label`, `Text`
- **Preprocessing**: The dataset undergoes label encoding, tokenization, and padding to prepare it for training.

## Model Architecture
- **Embedding Layer**: Converts text data into numerical vectors.
- **Bidirectional LSTM (64 units, return_sequences=True)**: Captures sequential dependencies in both forward and backward directions.
- **Dropout (0.5)**: Reduces overfitting.
- **Batch Normalization**: Normalizes activations for stable training.
- **Bidirectional LSTM (32 units)**: Further refines text representations.
- **Dropout (0.5)**: Additional regularization.
- **Dense (32 units, ReLU activation)**: Fully connected layer.
- **Dense (4 units, Softmax activation)**: Outputs class probabilities.

## Training & Evaluation
- **Loss Function**: Sparse Categorical Crossentropy (for multi-class classification)
- **Optimizer**: Adam (learning rate = 0.001)
- **Metrics**: Accuracy
- **Callbacks**:
  - `ReduceLROnPlateau`: Reduces learning rate if validation loss stagnates.

## Results
- The model is trained for 10 epochs with a batch size of 64.
- Final model accuracy is evaluated on the test set.
- A classification report is generated to assess precision, recall, and F1-score.
