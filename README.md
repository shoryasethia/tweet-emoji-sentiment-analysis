# Sentiment Analysis for Tweets containing Emojis

A comprehensive sentiment analysis system that combines text-based machine learning with emoji sentiment scoring to classify tweets as positive or negative.

## 🎯 Project Overview

This project implements a hybrid approach to tweet sentiment analysis by combining:
- **Text-based sentiment analysis** using logistic regression with TF-IDF features
- **Emoji sentiment analysis** using real-world emoji sentiment data
- **Hybrid fusion model** that intelligently combines both approaches

## 📊 Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | Notes |
|-------|----------|-----------|---------|----------|-------|
| Text-only | **78.49%** | 77.29% | 80.70% | 78.96% | Trained on full 1.6M tweets |
| Emoji-only | Variable | - | - | - | Depends on emoji presence |
| Hybrid Model | **Optimized** | Enhanced | Enhanced | Enhanced | Best performance on emoji-rich tweets |

### 🎉 Performance Highlights
- **Improved Accuracy**: Achieved 78.49% accuracy with full dataset training (up from 77.18%)
- **Balanced Performance**: Strong precision (77.29%) and recall (80.70%) scores
- **Robust F1-Score**: 78.96% F1-score demonstrates excellent overall performance
- **Full Dataset Utilization**: Now trained on complete 1.6M tweet dataset for maximum accuracy

## 🗂️ Project Structure

```
tweet-classification/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── models/                            # 🆕 Trained models directory
│   ├── tweet_sentiment_model.pkl      # Text-based sentiment model
│   ├── emoji_sentiment_map.pkl        # Emoji sentiment mapping
│   ├── hybrid_sentiment_model.pkl     # Optimized hybrid model
│   └── hybrid_model_metrics.pkl       # Performance metrics
├── data/                              # Dataset directory
│   ├── README.md                      # Data documentation
│   ├── tweets/                        # Sentiment140 dataset
│   │   ├── README.md                  # Download instructions
│   │   └── training.1600000.processed.noemoticon.csv
│   └── emoji/                         # Emoji sentiment data
│       ├── README.md                  # Download instructions
│       ├── Emoji_Sentiment_Data_v1.0.csv
│       ├── Emojitracker_20150604.csv
│       └── ESR_v1.0_format.txt
├── getdata.ipynb                      # Data download notebook
├── tweet_sentiment_analysis.ipynb    # Main training notebook
└── hybrid_sentiment_analysis.ipynb   # Hybrid model notebook
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd tweet-classification
pip install -r requirements.txt
```

### 2. Download Data

Follow the instructions in `data/README.md` to download the required datasets:
- Sentiment140 dataset (1.6M tweets)
- Emoji sentiment data (969 emojis)

### 3. Train Models

Run the notebooks in order:
1. `getdata.ipynb` - Download datasets
2. `tweet_sentiment_analysis.ipynb` - Train text-based model
3. `hybrid_sentiment_analysis.ipynb` - Create hybrid model

### 4. Use the Model

```python
import pickle

# Load the hybrid model
with open('models/hybrid_sentiment_model.pkl', 'rb') as f:
    hybrid_model = pickle.load(f)

# Predict sentiment
def predict_tweet(tweet):
    # Implementation from hybrid_sentiment_analysis.ipynb
    pass

# Example
tweet = "I love this movie! 😍❤️ It's amazing!"
result = predict_tweet(tweet)
print(f"Sentiment: {result['sentiment']}")
```

## 📈 Model Features

### Text-Based Analysis
- **Algorithm**: Logistic Regression with TF-IDF
- **Features**: 50k vocabulary, unigrams + bigrams
- **Preprocessing**: URL removal, mention/hashtag cleaning, emoji replacement
- **Training Data**: **Full 1.6M tweets** from Sentiment140 dataset
- **Performance**: 78.49% accuracy, 78.96% F1-score

### Emoji Analysis
- **Data Source**: Real social media emoji sentiment scores
- **Coverage**: 969 emojis with positive/negative/neutral labels
- **Scoring**: Normalized sentiment scores from -1 (negative) to +1 (positive)

### Hybrid Fusion
- **Weighting**: Adaptive weights based on emoji presence
- **Boost Factor**: Increased emoji influence when emojis are present
- **Fallback**: Text-only analysis when no emojis detected
- **Optimization**: Grid search for optimal parameter tuning

## 🛠️ Technical Details

### Model Architecture
1. **Text Pipeline**:
   - Text preprocessing (emoji replacement, cleaning)
   - TF-IDF vectorization (50k features, 1-2 grams)
   - Logistic regression classification

2. **Emoji Pipeline**:
   - Emoji extraction from text
   - Sentiment score lookup
   - Average sentiment calculation

3. **Hybrid Fusion**:
   - Weighted combination of text and emoji scores
   - Adaptive weighting based on emoji presence
   - Optimized parameters via grid search

### Performance Metrics
- **Accuracy**: 78.49% (primary evaluation metric)
- **Precision**: 77.29% (positive prediction accuracy)
- **Recall**: 80.70% (positive case detection rate)
- **F1-Score**: 78.96% (harmonic mean of precision and recall)
- **Training Scale**: Full 1.6M tweet dataset
- **Confusion Matrix**: Detailed error analysis available in notebooks

## 📝 Datasets

### Sentiment140
- **Source**: Stanford Sentiment140 dataset
- **Size**: 1.6 million tweets
- **Labels**: Binary (positive/negative)
- **Format**: CSV with sentiment, tweet text, metadata
- **Download**: Via Kaggle API (kazanova/sentiment140)

### Emoji Sentiment Data
- **Source**: Emoji sentiment research dataset
- **Size**: 969 unique emojis
- **Features**: Positive/negative/neutral occurrence counts
- **Format**: CSV with emoji, Unicode, sentiment scores
- **Download**: Via Kaggle API (thomasseleck/emoji-sentiment-data)

## 🔬 Notebooks

### 1. `getdata.ipynb`
- Downloads datasets from Kaggle
- Organizes data into proper directory structure
- Verifies data integrity

### 2. `tweet_sentiment_analysis.ipynb`
- Comprehensive text-based sentiment analysis
- Data preprocessing and feature engineering
- Model training and evaluation
- Feature importance analysis
- Model persistence

### 3. `hybrid_sentiment_analysis.ipynb`
- Hybrid model implementation
- Performance comparison (text vs emoji vs hybrid)
- Parameter optimization
- Detailed analysis of improvements
- Final model creation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
