from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from functools import wraps
import time

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='api_logs.log'
)
logger = logging.getLogger(__name__)


MODEL_PATH = "random_forest_model.joblib"
model = joblib.load(MODEL_PATH)


STATUS_MAPPING = {0: "Failure", 1: "Normal", 2: "Warning"}


prediction_cache = {}

def validate_features(required_features):
    """Decorator to validate incoming feature sets"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            if not data or "features" not in data:
                return jsonify({"error": "Invalid input. Provide 'features' in JSON format."}), 400
            
            missing_features = [feat for feat in required_features if feat not in data["features"]]
            if missing_features:
                return jsonify({
                    "error": f"Missing required features: {missing_features}",
                    "required_features": required_features
                }), 400
                
            return f(*args, **kwargs)
        return wrapper
    return decorator

def log_prediction(prediction_data):
    """Log prediction details"""
    logger.info(f"Prediction made: {prediction_data}")

def calculate_feature_importance(features):
    """Calculate and return feature importance"""
    importance_dict = dict(zip(features, model.feature_importances_))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

@app.route("/", methods=["GET"])
def home():
    """Enhanced home endpoint with model information"""
    return jsonify({
        "status": "active",
        "model_info": {
            "type": "Random Forest Classifier",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "features_supported": list(model.feature_names_in_),
            "classes": list(STATUS_MAPPING.values())
        },
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/health": "GET - Check API health",
            "/model-info": "GET - Get detailed model information"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    })

@app.route("/model-info", methods=["GET"])
def model_info():
    """Detailed model information endpoint"""
    return jsonify({
        "model_type": "Random Forest Classifier",
        "parameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
        },
        "features": list(model.feature_names_in_),
        "feature_importance": calculate_feature_importance(model.feature_names_in_),
        "classes": list(STATUS_MAPPING.values()),
        "last_trained": "2025-01-28"  
    })

@app.route("/predict", methods=["POST"])
@validate_features(model.feature_names_in_)
def predict():
    """Enhanced prediction endpoint"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        features = data["features"]
        feature_values = np.array([list(features.values())])
        pred_proba = model.predict_proba(feature_values)[0]
        predicted_class = int(np.argmax(pred_proba))
        confidence_scores = {STATUS_MAPPING[i]: float(prob) for i, prob in enumerate(pred_proba)}
        

        feature_importance = calculate_feature_importance(features.keys())
        
        response = {
            "prediction": {
                "class": predicted_class,
                "status": STATUS_MAPPING[predicted_class],
                "confidence_scores": confidence_scores,
                "threshold_warnings": []
            },
            "feature_analysis": {
                "importance": feature_importance,
                "anomalies": []
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "processing_time": f"{(time.time() - start_time):.3f}s",
                "model_version": "1.0"
            }
        }
        
        if features["Temperature"] > 40:
            response["prediction"]["threshold_warnings"].append("High temperature detected")
        if features["Signal_Strength"] < -85:
            response["prediction"]["threshold_warnings"].append("Low signal strength detected")
        
        log_prediction(response)
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Resource not found",
        "status_code": 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "status_code": 500
    }), 500

if __name__ != "vercel":
    if __name__ == "__main__":
        app.run(host="127.0.0.1", port=5000, debug=True)