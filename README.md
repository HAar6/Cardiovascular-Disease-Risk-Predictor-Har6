# Cardiovascular Diseases Prediction Analysis

A modern, interactive Streamlit web application for predicting cardiovascular diseases using Neural Networks. This application uses AI to analyze patient data and provide risk assessments for cardiovascular conditions.

## Features

- 🎯 **Simple & Clean UI**: User-friendly interface with modern design
- 🤖 **AI-Powered Predictions**: Neural network model trained on UCI Heart Disease dataset
- 📊 **Real-time Analysis**: Instant predictions with confidence scores
- 📋 **Comprehensive Input Form**: 13 medical parameters for accurate assessment
- ✅ **Binary Classification**: Clear results - Disease Detected or No Disease

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HAar6/Cardiovascular-Disease-Risk-Predictor-Har6.git
   cd Cardiovascular-Disease-Risk-Predictor-Har6
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Input Features

The application requires 13 patient parameters:

1. **Age** - Patient's age in years
2. **Sex** - Gender (Male/Female)
3. **Chest Pain Type** - Type of chest pain experienced
4. **Resting Blood Pressure** - Blood pressure at rest (mm Hg)
5. **Serum Cholesterol** - Cholesterol level (mg/dl)
6. **Fasting Blood Sugar** - > 120 mg/dl (Yes/No)
7. **Resting ECG** - Electrocardiogram results
8. **Maximum Heart Rate** - Peak heart rate achieved
9. **Exercise Induced Angina** - Yes/No
10. **ST Depression** - ST depression value
11. **Slope of Peak Exercise ST** - Slope type
12. **Number of Major Vessels** - 0 to 3
13. **Thalassemia** - Type of thalassemia

## How It Works

1. Enter all patient details in the form
2. Click "Get Cardiovascular Disease Prediction" button
3. View the prediction result with confidence score
4. Get recommendations based on the prediction

## Model Information

- **Dataset**: UCI Machine Learning Repository - Cleveland Heart Disease Dataset
- **Model Type**: Neural Network (Binary Classification)
- **Architecture**: 
  - Input Layer: 13 features
  - Hidden Layer 1: 8 neurons (ReLU activation)
  - Hidden Layer 2: 4 neurons (ReLU activation)
  - Output Layer: 1 neuron (Sigmoid activation)
- **Accuracy**: ~78% on test set
- **Training**: 100 epochs with Adam optimizer

## Project Structure

```
Cardiovascular-Disease-Risk-Predictor-Har6/
├── app.py                 # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── SETUP.md             # Setup instructions
├── GITHUB_UPLOAD.md     # GitHub upload guide
└── .gitignore           # Git ignore file
```

## Requirements

- Python 3.7+
- Streamlit >= 1.20.0
- TensorFlow >= 2.5.0, < 2.8.0
- Pandas >= 1.1.0, < 1.2.0
- NumPy >= 1.19.0, < 1.20.0
- Scikit-learn >= 0.24.0, < 1.0.0

## Medical Disclaimer

⚠️ **IMPORTANT**: This prediction tool is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding a medical condition.

## License

This project is open source and available for educational purposes.

## Contributing

Contributions, issues, and feature requests are welcome!

## Author

Created for cardiovascular disease risk assessment using machine learning.
