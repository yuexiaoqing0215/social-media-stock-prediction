```
# Sentiment-Aware Stock Movement Prediction with LSTM and Attention

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![NLP](https://img.shields.io/badge/NLP-FinBERT-yellowgreen)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 📖 Abstract

This project proposes a **sentiment-aware deep learning framework** for predicting stock price movements. By integrating **Reddit-derived sentiment signals** into an **LSTM model enhanced with an attention mechanism**, we achieve up to **68% accuracy** in predicting the next day's price direction for Tesla stock (Mar 2024 - Mar 2025). The model outperforms traditional price-only models, demonstrating the value of social media sentiment in financial forecasting.

## 🚀 Key Features

- **Dual-Modality Input**: Combines historical stock prices with daily aggregated sentiment scores from Reddit.
- **Attention Mechanism**: Dynamically weights the importance of sentiment and price features over time.
- **Comparative Analysis**: Benchmarks against multiple baselines (LSTM, Random Forest, Logistic Regression, ARIMA, Naïve).
- **Interactive Dashboard**: Built with Streamlit for visualizing predictions and model performance.
- **End-to-End Pipeline**: From data collection and sentiment analysis to model training and evaluation.

## 📊 Results Summary

| Model | Test Accuracy |
|--------|----------------|
| Sentiment-Aware LSTM (Ours) | **67.92%** |
| Baseline LSTM | 58.49% |
| Naïve Baseline | 51.92% |
| ARIMA | 50.94% |
| Logistic Regression | 39.62% |
| Random Forest | 39.62% |

The sentiment-aware model significantly outperforms all baselines, highlighting the benefit of incorporating sentiment data via attention.

## 🗂️ Project Structure

```
.
├── A11_data/                         # Data directory
│   ├── data/
│   │   ├── A0_reddit_zst/            # Raw Reddit data (compressed)
│   │   ├── A1_top100_posts/          # Filtered top 100 daily posts
│   │   ├── A2_sentiment_analysis/    # FinBERT sentiment labels
│   │   ├── A3_daily_sentiment_score/ # Daily weighted sentiment scores
│   │   └── B1_stock_data/            # Historical stock data (TSLA)
├── code/
│   ├── A1_extract_top100_posts.ipynb      # Reddit data preprocessing
│   ├── A2_sentiment_analysis.ipynb        # FinBERT sentiment analysis
│   ├── A3_daily_sentiment_score.ipynb     # Daily sentiment aggregation
│   ├── B1_stock_data.ipynb                # Stock data retrieval
│   ├── C_stock_price_forecasting.ipynb    # LSTM regression models
│   ├── D_directional_forecasting.ipynb    # LSTM classification models
│   └── E_dashboard.py                     # Streamlit dashboard
├── outputs/                          # Trained models & results
├── README.md                         # Project overview (this file)
└── requirements.txt                  # Python dependencies
```

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sentiment-stock-prediction.git
cd sentiment-stock-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
Execute the notebooks in the following order to reproduce the results:

1. `A1_extract_top100_posts.ipynb` – Preprocess Reddit data
2. `A2_sentiment_analysis.ipynb` – Run FinBERT sentiment analysis
3. `A3_daily_sentiment_score.ipynb` – Compute daily sentiment scores
4. `B1_stock_data.ipynb` – Download Tesla stock data
5. `C_stock_price_forecasting.ipynb` – Train LSTM regression models
6. `D_directional_forecasting.ipynb` – Train sentiment-aware LSTM classifier

### 4. Launch Dashboard
```bash
streamlit run code/E_dashboard.py
```

## Methodology

### Data Collection
- **Stock Data**: Tesla (TSLA) daily OHLCV prices from Yahoo Finance (`yfinance`), Mar 2024 – Mar 2025.
- **Sentiment Data**: Reddit post titles from Academic Torrent, filtered for "Tesla", top 100 posts per day by upvotes.

### Sentiment Analysis
- **Model**: FinBERT (ProsusAI implementation) for financial text sentiment classification.
- **Output**: Positive (+1), Negative (-1), Neutral (0) labels.
- **Aggregation**: Daily weighted sentiment score based on upvotes.

### Model Architecture
- **Baseline LSTM**: Uses only historical closing prices.
- **Sentiment-Aware LSTM**: Dual-stream LSTM with attention:
  - Price LSTM: Encodes past 7 days of closing prices.
  - Sentiment LSTM: Encodes past 7 days of sentiment scores.
  - Attention Layer: Weights important time steps.
  - Fully Connected Classifier: Predicts up/down movement.

## Limitations

- Sentiment data sourced only from Reddit (no Twitter/News).
- Only post titles analyzed (excludes comments).
- Limited to one year of data (Mar 2024 – Mar 2025).
- Model trained only on Tesla stock; generalizability untested.

## Future Work

- Incorporate multi-platform sentiment (Twitter, news, YouTube).
- Extend time horizon and include technical indicators.
- Experiment with Transformer-based models (e.g., BERT, FinBERT).
- Apply transfer learning and model ensembling.

## Contributors

- **Cheng Wing Sze (Celia)** – Baseline models, exploratory analysis, Streamlit dashboard.
- **Hongchao Wang** – LSTM model development, report consolidation.
- **Woo Shirong (Ava)** – Data collection, literature review, project management.
- **Xiaoqing Yue** – Data preprocessing, sentiment analysis.
- **Xiyue Wang** – Limitations, conclusion, literature review.

## Acknowledgments

We thank Professor Nan Ye for his guidance and encouragement throughout the project.

## License

This project is for academic purposes. Please cite the original authors of the data and models used (FinBERT, Yahoo Finance, Reddit via Academic Torrent).

## Contact

For questions or collaborations, feel free to open an issue or contact yuexiaoqing0215@gmail.com.
