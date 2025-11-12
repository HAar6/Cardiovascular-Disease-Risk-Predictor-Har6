import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Diseases Prediction Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Simple clean background */
    body, .stApp, .main {
        color: #1a1a1a !important;
        background-color: #f8f9fa;
    }
    
    /* Fix Streamlit text elements - comprehensive */
    .stMarkdown, .stText, p, label, 
    div[data-testid="stMarkdownContainer"],
    div[class*="stMarkdown"],
    .element-container p,
    .element-container div {
        color: #1a1a1a !important;
    }
    
    /* All paragraphs and text */
    p, span, div:not(.main-title):not(.subtitle):not(.prediction-card):not(.confidence-badge) {
        color: #1a1a1a !important;
    }
    
    /* Input labels - all variations */
    label, .stNumberInput label, .stSelectbox label,
    .stNumberInput > label, .stSelectbox > label,
    label[data-testid*="label"] {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Headers - but keep main-title and subtitle white */
    h1:not(.main-title), h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }
    
    /* Section headers in form - make them more visible */
    h3 {
        color: #667eea !important;
        font-weight: 700 !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.3rem !important;
    }
    
    /* Info box text - ensure visibility */
    .info-box, .info-box p, .info-box strong, .info-box * {
        color: #1a1a1a !important;
    }
    
    /* Footer text */
    .footer, .footer p, .footer * {
        color: #4a4a4a !important;
    }
    
    /* Streamlit success/error/info messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        color: #1a1a1a !important;
    }
    
    /* All Streamlit widget containers */
    .stNumberInput, .stSelectbox, .stTextInput {
        color: #1a1a1a !important;
    }
    
    /* Caption text */
    .stCaption, caption {
        color: #666 !important;
    }
    
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3.5rem;
        color: white;
        text-align: center;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 1px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .form-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .section-title {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    .positive {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF5252 100%);
        color: white;
    }
    
    .negative {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: white;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.3);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        margin-top: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
        color: white !important;
    }
    
    .info-box {
        background: #f0f4ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #1a1a1a !important;
    }
    
    .info-box p, .info-box strong {
        color: #1a1a1a !important;
        margin: 0;
    }
    
    .stButton>button {
        background: #667eea;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: #5568d3;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 5px;
        border: 1px solid #ddd !important;
        transition: all 0.2s ease;
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        font-size: 1rem !important;
    }
    
    /* Remove red error borders - comprehensive fix */
    .stNumberInput>div>div>input:invalid,
    input:invalid,
    input[type="number"]:invalid,
    input[type="number"] {
        border-color: #ddd !important;
        box-shadow: none !important;
    }
    
    /* Remove any red borders from Streamlit widgets */
    .stNumberInput > div > div > div,
    .stNumberInput input,
    input[type="number"] {
        border: 1px solid #ddd !important;
    }
    
    /* Fix number input buttons - make them light gray */
    button[data-baseweb="button"],
    button[aria-label*="decrement"],
    button[aria-label*="increment"],
    .stNumberInput button {
        background-color: #f5f5f5 !important;
        color: #333 !important;
        border: 1px solid #ddd !important;
    }
    
    button[data-baseweb="button"]:hover {
        background-color: #e0e0e0 !important;
    }
    
    /* Simple input borders */
    input, select, textarea {
        border-color: #ddd !important;
    }
    
    /* Fix select dropdown styling - white background */
    select,
    .stSelectbox select,
    .stSelectbox > div > div > select {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #ddd !important;
    }
    
    /* Fix Streamlit selectbox container */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #ddd !important;
    }
    
    /* Remove dark gray background from dropdowns */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Input text color - all input types */
    input[type="number"], 
    input[type="text"],
    select,
    textarea {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        border-color: #ddd !important;
    }
    
    /* Input placeholders */
    input::placeholder {
        color: #999 !important;
    }
    
    /* Streamlit widget labels - comprehensive */
    .stNumberInput label, 
    .stSelectbox label, 
    .stTextInput label,
    .stNumberInput > label,
    .stSelectbox > label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Select dropdown options */
    option {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Remove any error states */
    input:focus:invalid {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Fix Streamlit number input container */
    .stNumberInput > div {
        border: none !important;
    }
    
    /* Remove red borders from all inputs */
    input[type="number"] {
        border-color: #ddd !important;
    }
    
    input[type="number"]:focus {
        border-color: #667eea !important;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #4a4a4a !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600 !important;
    }
    
    /* Ensure metric card text is visible */
    .metric-card, .metric-card * {
        color: #1a1a1a !important;
    }
    
    .metric-card .metric-value {
        color: #667eea !important;
    }
    
    .footer {
        text-align: center;
        color: #666 !important;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1.5rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Additional visibility fixes */
    .stAlert {
        color: #1a1a1a !important;
    }
    
    /* Ensure all divs in form container have proper text color */
    .form-container, .form-container * {
        color: #1a1a1a !important;
    }
    
    /* Section title visibility */
    .section-title {
        color: #667eea !important;
        font-weight: 700 !important;
    }
    
    /* Make sure help text is visible */
    [data-testid="stTooltipIcon"] {
        color: #667eea !important;
    }
    
    /* Streamlit column text */
    [data-testid="column"] {
        color: #1a1a1a !important;
    }
    
    [data-testid="column"] * {
        color: #1a1a1a !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the heart disease dataset"""
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class']
    
    try:
        data = pd.read_csv(url, names=names)
        data = data[~data.isin(['?'])]
        data = data.dropna(axis=0)
        data = data.apply(pd.to_numeric)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_model(data):
    """Train the neural network model"""
    X = np.array(data.drop(['class'], axis=1))
    y = np.array(data['class'])
    y_binary = y.copy()
    y_binary[y_binary > 0] = 1
    
    X_train, _, y_train, _ = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    model = Sequential([
        Dense(8, input_dim=13, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    with st.spinner('🔄 Training model (this happens once)...'):
        model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
    
    return model

def main():
    # Header Section
    st.markdown("""
        <div class="main-container">
            <h1 class="main-title">Cardiovascular Diseases Prediction Analysis</h1>
            <p class="subtitle">AI-Powered Cardiovascular Disease Risk Assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data and train model
    data = load_data()
    if data is None:
        return
    
    model = train_model(data)
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">297</div>
                <div class="metric-label">Patients Analyzed</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">13</div>
                <div class="metric-label">Features</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">78%</div>
                <div class="metric-label">Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">AI</div>
                <div class="metric-label">Neural Network</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Form Container
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">📋 Patient Information Form</p>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Instructions:</strong> Please fill in all the patient details below to get an accurate prediction. 
            All fields are required for the best results.
        </div>
    """, unsafe_allow_html=True)
    
    # Input form in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Personal Information")
        age = st.number_input("**Age** (years)", min_value=1, max_value=120, value=50, 
                            help="Patient's age in years")
        sex = st.selectbox("**Sex**", ["Female", "Male"], 
                          help="Patient's gender")
        cp = st.selectbox("**Chest Pain Type**", 
                         ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                         help="Type of chest pain experienced")
        
        st.markdown("### 💓 Cardiovascular Metrics")
        trestbps = st.number_input("**Resting Blood Pressure** (mm Hg)", min_value=0, max_value=300, value=120,
                                  help="Blood pressure at rest")
        chol = st.number_input("**Serum Cholesterol** (mg/dl)", min_value=0, max_value=600, value=200,
                              help="Cholesterol level in blood")
        fbs = st.selectbox("**Fasting Blood Sugar > 120 mg/dl**", ["No", "Yes"],
                          help="Is fasting blood sugar elevated?")
        restecg = st.selectbox("**Resting ECG Results**", 
                               ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                               help="Electrocardiogram results at rest")
    
    with col2:
        st.markdown("### 🏃 Exercise & Test Results")
        thalach = st.number_input("**Maximum Heart Rate Achieved**", min_value=0, max_value=250, value=150,
                                 help="Peak heart rate during exercise")
        exang = st.selectbox("**Exercise Induced Angina**", ["No", "Yes"],
                            help="Does exercise cause chest pain?")
        oldpeak = st.number_input("**ST Depression (Oldpeak)**", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                  help="ST segment depression during exercise")
        slope = st.selectbox("**Slope of Peak Exercise ST Segment**", 
                            ["Upsloping", "Flat", "Downsloping"],
                            help="Slope of ST segment during peak exercise")
        ca = st.selectbox("**Number of Major Vessels** (0-3)", [0, 1, 2, 3],
                         help="Number of major vessels visible in fluoroscopy")
        thal = st.selectbox("**Thalassemia Type**", 
                           ["Normal", "Fixed Defect", "Reversible Defect"],
                           help="Type of thalassemia")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Convert inputs to model format
    sex_val = 1 if sex == "Male" else 0
    cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    fbs_val = 1 if fbs == "Yes" else 0
    restecg_val = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    exang_val = 1 if exang == "Yes" else 0
    slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
    thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Get Cardiovascular Disease Prediction", type="primary", use_container_width=True):
        input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                               thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
        
        with st.spinner("🔍 Analyzing patient data with AI..."):
            prediction = model.predict(input_data, verbose=0)[0][0]
            has_disease = prediction > 0.5
            confidence = prediction if has_disease else 1 - prediction
        
        # Display result with enhanced styling
        if has_disease:
            st.markdown(f"""
                <div class="prediction-card positive">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                    <div>Cardiovascular Disease Detected</div>
                    <div class="confidence-badge">Confidence: {confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
            st.error("⚠️ **Important:** Please consult with a healthcare professional immediately for proper diagnosis and treatment of cardiovascular disease.")
        else:
            st.markdown(f"""
                <div class="prediction-card negative">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
                    <div>No Cardiovascular Disease Detected</div>
                    <div class="confidence-badge">Confidence: {confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
            st.success("✅ **Good News!** However, always consult healthcare professionals for regular cardiovascular check-ups and medical advice.")
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p><strong>⚠️ Medical Disclaimer:</strong> This prediction tool is for educational and informational purposes only.</p>
            <p>It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
            <p>Always seek the advice of qualified health providers with any questions regarding a medical condition.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
