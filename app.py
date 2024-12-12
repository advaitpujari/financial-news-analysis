import os
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

app = Flask(__name__)

# Load pre-trained model and tokenizer
MODEL_PATH = 'model/financial_news_model'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Load additional supporting files
stocks_data = pd.read_csv('data/stock_sectors.csv')
impact_lookup = joblib.load('model/impact_lookup.pkl')

def preprocess_text(text):
    """Preprocess input text for model analysis"""
    # Implement text cleaning, lowercasing, etc.
    return text

def predict_stock_impact(news_text):
    """Analyze news text and predict stock/sector impacts"""
    # Tokenize and encode the text
    inputs = tokenizer(news_text, return_tensors='tf', padding=True, truncation=True)
    
    # Get model predictions
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    
    # Process predictions to extract stocks, sectors, and impact levels
    affected_stocks = []
    affected_sectors = []
    impact_levels = {}
    
    # Implement logic to map model predictions to stocks and sectors
    # This is a placeholder and would need a sophisticated implementation
    
    return {
        'affected_stocks': affected_stocks,
        'affected_sectors': affected_sectors,
        'impact_levels': impact_levels
    }

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    """Endpoint for news analysis"""
    news_text = request.form.get('news_text', '')
    
    if not news_text:
        return jsonify({
            'error': 'No news text provided',
            'status': 'failure'
        }), 400
    
    try:
        results = predict_stock_impact(news_text)
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failure'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
