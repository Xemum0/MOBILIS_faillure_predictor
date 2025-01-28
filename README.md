# Random Forest Prediction API

## Overview
The **Random Forest Prediction API** is a Flask-based application for making predictions using a pre-trained Random Forest model. It provides endpoints to make predictions, check API health, and retrieve model information, along with robust logging and input validation.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Endpoints](#endpoints)
- [Input Validation](#input-validation)
- [Feature Analysis](#feature-analysis)
- [Error Handling](#error-handling)
- [Logging](#logging)
- [License](#license)

---

## Getting Started

### Requirements
- Python 3.8+
- Flask
- Scikit-learn
- Numpy
- Pandas

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the trained Random Forest model file (`random_forest_model.joblib`) in the root directory.

4. Start the API:
   ```bash
   python app.py
   ```

---

## Endpoints

### 1. **Home**
- **URL**: `/`
- **Method**: GET
- **Description**: Returns API status and model information.
- **Response**:
  ```json
  {
      "status": "active",
      "model_info": {
          "type": "Random Forest Classifier",
          "n_estimators": 100,
          "max_depth": 10,
          "features_supported": ["Temperature", "Signal_Strength", ...],
          "classes": ["Failure", "Normal", "Warning"]
      },
      "endpoints": {
          "/predict": "POST - Make predictions",
          "/health": "GET - Check API health",
          "/model-info": "GET - Get detailed model information"
      }
  }
  ```

### 2. **Health Check**
- **URL**: `/health`
- **Method**: GET
- **Description**: Checks API health status.
- **Response**:
  ```json
  {
      "status": "healthy",
      "timestamp": "2025-01-28T10:00:00",
      "model_loaded": true
  }
  ```

### 3. **Model Information**
- **URL**: `/model-info`
- **Method**: GET
- **Description**: Provides detailed information about the model.
- **Response**:
  ```json
  {
      "model_type": "Random Forest Classifier",
      "parameters": {
          "n_estimators": 100,
          "max_depth": 10,
          "min_samples_split": 2
      },
      "features": ["Temperature", "Signal_Strength", ...],
      "feature_importance": {
          "Temperature": 0.35,
          "Signal_Strength": 0.25
      },
      "classes": ["Failure", "Normal", "Warning"],
      "last_trained": "2025-01-28"
  }
  ```

### 4. **Prediction**
- **URL**: `/predict`
- **Method**: POST
- **Description**: Makes predictions based on input features.
- **Request Body**:
  ```json
  {
      "features": {
          "Temperature": 30,
          "Signal_Strength": -75,
          ...
      }
  }
  ```
- **Response**:
  ```json
  {
      "prediction": {
          "class": 1,
          "status": "Normal",
          "confidence_scores": {
              "Failure": 0.1,
              "Normal": 0.8,
              "Warning": 0.1
          },
          "threshold_warnings": ["High temperature detected"]
      },
      "feature_analysis": {
          "importance": {
              "Temperature": 0.35,
              "Signal_Strength": 0.25
          },
          "anomalies": []
      },
      "metadata": {
          "timestamp": "2025-01-28T10:00:00",
          "processing_time": "0.123s",
          "model_version": "1.0"
      }
  }
  ```

---

## Input Validation
The API validates the request body to ensure required features are present. Missing features will return an error response:
```json
{
    "error": "Missing required features: ['Temperature', 'Signal_Strength']",
    "required_features": ["Temperature", "Signal_Strength", ...]
}
```

---

## Feature Analysis
- **Feature Importance**: Calculates the relative importance of each feature in the model.
- **Threshold Warnings**: Alerts for abnormal feature values (e.g., high temperature, low signal strength).

---

## Error Handling
The API handles common errors gracefully:
- **404 Not Found**: Resource not found.
- **500 Internal Server Error**: General server errors.

Example response for a 404 error:
```json
{
    "error": "Resource not found",
    "status_code": 404
}
```

---

## Logging
- Logs all predictions and errors to `api_logs.log`.
- Example log entry:
  ```
  2025-01-28 10:00:00,123 - __main__ - INFO - Prediction made: {"prediction": {"class": 1, ...}}
  ```

---

## License
This project is licensed under the MIT License.
