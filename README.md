# SvaraAI Reply Classifier

An AI-powered email reply classification system that categorizes prospect responses as positive, negative, or neutral to help sales teams prioritize their outreach efforts.

## Project Overview

This project implements a complete machine learning pipeline for classifying email replies, including:
- Data preprocessing and feature engineering
- Multiple baseline model training (Logistic Regression, LightGBM, Random Forest, Naive Bayes)
- Transformer model fine-tuning (DistilBERT)
- FastAPI deployment with REST endpoints
- Comprehensive performance evaluation

## Performance Results

| Model | Test Accuracy | Test F1 Score | Training Time |
|-------|---------------|---------------|---------------|
| **Logistic Regression** | **95.38%** | **95.35%** | 0.02s |
| DistilBERT | 93.85% | 93.78% | 662s |
| Random Forest | 89.23% | 89.32% | 0.67s |
| Naive Bayes | 90.77% | 90.71% | 0.02s |
| LightGBM | 66.15% | 65.82% | 0.21s |

**Production Recommendation**: Logistic Regression (best performance + fastest inference)

## Project Structure

```
├── notebook.ipynb              # Main ML pipeline implementation
├── production_api.py           # FastAPI application
├── answers.md                  # Technical reasoning (Part C)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container deployment
├── logistic_regression_model.pkl    # Trained model
└── tfidf_vectorizer.pkl        # Feature vectorizer
```

## Setup Instructions

### Google Colab (Recommended)

1. **Upload your dataset**: Place `replies.csv` in your Colab environment
2. **Run the pipeline**: Execute the notebook cells step by step
3. **Start the API**: The FastAPI server will start automatically

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the ML pipeline**:
   ```bash
   # Upload your notebook to Jupyter and run all cells
   jupyter notebook notebook.ipynb
   ```

3. **Start the API server**:
   ```bash
   python production_api.py
   ```

### Docker Deployment

1. **Build the container**:
   ```bash
   docker build -t svaraai-classifier .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 svaraai-classifier
   ```

## API Usage

### Start the API
The FastAPI server runs on `http://localhost:8000`

### API Endpoints

- **GET /** - API information and health status
- **GET /health** - Health check endpoint
- **GET /stats** - Model performance statistics
- **POST /predict** - Single text classification
- **POST /batch_predict** - Batch text classification

### Example Usage

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict"      -H "Content-Type: application/json"      -d '{"text": "Looking forward to the demo!"}'
```

**Response:**
```json
{
  "label": "positive",
  "confidence": 0.682
}
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/batch_predict"      -H "Content-Type: application/json"      -d '{"texts": ["I am interested!", "Not for us", "Let me think about it"]}'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This looks perfect for our needs!"}
)
result = response.json()
print(f"Label: {result['label']}, Confidence: {result['confidence']}")
```

## Model Details

### Best Model: Logistic Regression
- **Accuracy**: 95.38%
- **F1 Score**: 95.35%
- **Features**: TF-IDF vectorization with 211 features
- **Training Time**: 0.02 seconds
- **Inference Time**: ~0.005 seconds per request
- **Throughput**: ~185 requests/second

### Label Categories
- **Positive**: Interested, wants demo, ready to proceed
- **Negative**: Not interested, rejection, already have solution  
- **Neutral**: Need more info, will think about it, non-committal

## Performance Metrics

The API demonstrates excellent performance:
- Average response time: 0.005 seconds
- 100% successful request rate in testing
- Handles both single and batch predictions
- Built-in error handling and validation

## Dataset Information

- **Total Samples**: 2,129 email replies
- **Training Set**: 256 samples (80%)
- **Test Set**: 65 samples (20%)
- **Features**: TF-IDF vectorized text with n-grams (1,2)
- **Classes**: Balanced distribution across positive, negative, neutral

## Dependencies

- fastapi>=0.68.0
- uvicorn>=0.15.0
- scikit-learn>=1.1.0
- pandas>=1.4.0
- numpy>=1.21.0
- transformers>=4.21.0
- torch>=1.12.0

## Production Deployment

The model is production-ready with:
- FastAPI web framework for scalable deployment
- Docker containerization support
- Comprehensive error handling and logging
- Input validation and sanitization
- Health monitoring endpoints
- Batch processing capabilities

For production deployment, consider:
- Load balancing for high traffic
- Database logging of predictions
- Model monitoring and drift detection
- A/B testing framework for model updates







# SvaraAI Reply Classifier

## Model Files Setup

### Option 1: Use Pre-trained Models
If you have the model files:
1. Place `logistic_regression_model.pkl` and `tfidf_vectorizer.pkl` in the project directory
2. Run the API normally

### Option 2: Train Models from Notebook
If model files are missing:
1. Open the provided Jupyter notebook
2. Run all cells in order, especially:
   - Step 1: Data Loading
   - Step 2: Data Preprocessing  
   - Step 3: Baseline Model Training
3. The models will be automatically saved to your Google Drive
4. Download the `.pkl` files and place them in the project directory

### Option 3: Demo Mode
The API includes a demo mode that works without model files:
- Uses rule-based predictions for demonstration
- Automatically activates when model files are missing
- Perfect for testing the API structure

## Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd svaraai-classifier

# Install dependencies
pip install -r requirements.txt

# Run the API (works with or without model files)
python production_api.py


## Troubleshooting

**Common Issues:**

1. **Port already in use**: Change port in `production_api.py`
2. **Model files missing**: Ensure .pkl files are in the same directory
3. **Dependencies error**: Run `pip install -r requirements.txt`

**For Google Colab:**
- Models are automatically saved to the session
- Use `nest_asyncio` for running FastAPI
- Server runs in background thread

## License

This project is created for the SvaraAI AI/ML Engineer Internship Assignment.

**Author**: Mohammad Sameer

**Date**: 24 September 2025  
**Contact**: 6304009764
