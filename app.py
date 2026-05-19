import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import altair as alt
import warnings
warnings.filterwarnings('ignore')
# Initialize session state for theme
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Diseases Prediction Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme CSS Variables based on session state
theme_css = """
    :root {
        --bg-color: #f8f9fa;
        --text-color: #1a1a1a;
        --form-bg: #ffffff;
        --form-border: #dddddd;
        --metric-card-bg: #ffffff;
        --metric-card-text: #1a1a1a;
        --metric-label-color: #4a4a4a;
        --info-bg: #f0f4ff;
        --info-border: #667eea;
        --input-bg: #ffffff;
        --input-border: #dddddd;
        --input-text: #1a1a1a;
        --footer-bg: #ffffff;
        --footer-text: #666666;
        --accent-color: #667eea;
    }
""" if st.session_state.theme == "light" else """
    :root {
        --bg-color: #0f172a;
        --text-color: #f8fafc;
        --form-bg: #1e293b;
        --form-border: #334155;
        --metric-card-bg: #1e293b;
        --metric-card-text: #f8fafc;
        --metric-label-color: #94a3b8;
        --info-bg: #1e1b4b;
        --info-border: #818cf8;
        --input-bg: #0f172a;
        --input-border: #334155;
        --input-text: #f8fafc;
        --footer-bg: #1e293b;
        --footer-text: #94a3b8;
        --accent-color: #818cf8;
    }
"""

st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

# Enhanced Custom Styling using CSS variables
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Simple clean background */
    body, .stApp, .main {
        color: var(--text-color) !important;
        background-color: var(--bg-color) !important;
    }
    
    /* Fix Streamlit text elements - comprehensive */
    .stMarkdown, .stText, p, label, 
    div[data-testid="stMarkdownContainer"],
    div[class*="stMarkdown"],
    .element-container p,
    .element-container div {
        color: var(--text-color) !important;
    }
    
    /* All paragraphs and text */
    p, span, div:not(.main-title):not(.subtitle):not(.prediction-card):not(.confidence-badge) {
        color: var(--text-color) !important;
    }
    
    /* Input labels - all variations */
    label, .stNumberInput label, .stSelectbox label,
    .stNumberInput > label, .stSelectbox > label,
    label[data-testid*="label"] {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Headers - but keep main-title and subtitle white */
    h1:not(.main-title), h2, h3:not(.section-header), h4, h5, h6 {
        color: var(--text-color) !important;
    }
    
    /* Section headers in form - make them more visible */
    h3 {
        color: var(--accent-color) !important;
        font-weight: 700 !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.3rem !important;
    }
    
    /* Info box text - ensure visibility */
    .info-box, .info-box p, .info-box strong, .info-box * {
        color: var(--text-color) !important;
    }
    
    /* Footer text */
    .footer, .footer p, .footer * {
        color: var(--footer-text) !important;
    }
    
    /* Streamlit success/error/info messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        color: var(--text-color) !important;
    }
    
    /* All Streamlit widget containers */
    .stNumberInput, .stSelectbox, .stTextInput {
        color: var(--text-color) !important;
    }
    
    /* Caption text */
    .stCaption, caption {
        color: var(--footer-text) !important;
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
        background: var(--form-bg);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 1px solid var(--form-border);
    }
    
    .section-title {
        font-size: 1.5rem;
        color: var(--accent-color);
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--accent-color);
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
        background: var(--info-bg);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--info-border);
        margin: 1rem 0;
        color: var(--text-color) !important;
    }
    
    .info-box p, .info-box strong {
        color: var(--text-color) !important;
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
        border: 1px solid var(--input-border) !important;
        transition: all 0.2s ease;
        color: var(--input-text) !important;
        background-color: var(--input-bg) !important;
        font-size: 1rem !important;
    }
    
    /* Remove red error borders - comprehensive fix */
    .stNumberInput>div>div>input:invalid,
    input:invalid,
    input[type="number"]:invalid,
    input[type="number"] {
        border-color: var(--input-border) !important;
        box-shadow: none !important;
    }
    
    /* Remove any red borders from Streamlit widgets */
    .stNumberInput > div > div > div,
    .stNumberInput input,
    input[type="number"] {
        border: 1px solid var(--input-border) !important;
    }
    
    /* Fix number input buttons */
    button[data-baseweb="button"],
    button[aria-label*="decrement"],
    button[aria-label*="increment"],
    .stNumberInput button {
        background-color: var(--form-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--input-border) !important;
    }
    
    button[data-baseweb="button"]:hover {
        background-color: var(--input-bg) !important;
    }
    
    /* Simple input borders */
    input, select, textarea {
        border-color: var(--input-border) !important;
    }
    
    /* Fix select dropdown styling */
    select,
    .stSelectbox select,
    .stSelectbox > div > div > select {
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border: 1px solid var(--input-border) !important;
    }
    
    /* Fix Streamlit selectbox container */
    .stSelectbox > div > div {
        background-color: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
    }
    
    /* Remove dark gray background from dropdowns */
    [data-baseweb="select"] {
        background-color: var(--input-bg) !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
    }
    
    /* Style the selectbox dropdown menu (popover portal) */
    [data-baseweb="popover"] {
        background-color: var(--input-bg) !important;
        border: 1px solid var(--form-border) !important;
        border-radius: 8px !important;
    }
    
    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] [role="option"] {
        background-color: var(--input-bg) !important;
        color: var(--text-color) !important;
    }
    
    /* Set default background/text for all children in the popover */
    [data-baseweb="popover"] * {
        background-color: var(--input-bg) !important;
        color: var(--text-color) !important;
    }
    
    /* Highlight hovered/selected options */
    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] li[aria-selected="true"],
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: #ffffff !important;
    }
    
    /* Highlight all children of hovered/selected options */
    [data-baseweb="popover"] li:hover *,
    [data-baseweb="popover"] li[aria-selected="true"] *,
    [data-baseweb="popover"] [role="option"]:hover *,
    [data-baseweb="popover"] [role="option"][aria-selected="true"] * {
        background-color: var(--accent-color) !important;
        color: #ffffff !important;
    }

    
    /* Input text color - all input types */
    input[type="number"], 
    input[type="text"],
    select,
    textarea {
        color: var(--input-text) !important;
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
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
        color: var(--text-color) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Select dropdown options */
    option {
        color: var(--input-text) !important;
        background-color: var(--input-bg) !important;
    }
    
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Remove any error states */
    input:focus:invalid {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Fix Streamlit number input container */
    .stNumberInput > div {
        border: none !important;
    }
    
    /* Remove red borders from all inputs */
    input[type="number"] {
        border-color: var(--input-border) !important;
    }
    
    input[type="number"]:focus {
        border-color: var(--accent-color) !important;
    }
    
    .metric-card {
        background: var(--metric-card-bg);
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
        border: 1px solid var(--form-border);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-color);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--metric-label-color) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600 !important;
    }
    
    /* Ensure metric card text is visible */
    .metric-card, .metric-card * {
        color: var(--metric-card-text) !important;
    }
    
    .metric-card .metric-value {
        color: var(--accent-color) !important;
    }
    
    .footer {
        text-align: center;
        color: var(--footer-text) !important;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1.5rem;
        background: var(--footer-bg);
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid var(--form-border);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Additional visibility fixes */
    .stAlert {
        color: var(--text-color) !important;
    }
    
    /* Ensure all divs in form container have proper text color */
    .form-container, .form-container * {
        color: var(--text-color) !important;
    }
    
    /* Section title visibility */
    .section-title {
        color: var(--accent-color) !important;
        font-weight: 700 !important;
    }
    
    /* Make sure help text is visible */
    [data-testid="stTooltipIcon"] {
        color: var(--accent-color) !important;
    }
    
    /* Streamlit column text */
    [data-testid="column"] {
        color: var(--text-color) !important;
    }
    
    [data-testid="column"] * {
        color: var(--text-color) !important;
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = MLPClassifier(
        hidden_layer_sizes=(8, 4),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        batch_size=10,
        random_state=42
    )
    
    with st.spinner('🔄 Training neural network model...'):
        model.fit(X_train_scaled, y_train)
        
    # Calculate test metrics
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate feature correlations with the binary class
    df_corr = data.copy()
    df_corr['class_binary'] = df_corr['class'].apply(lambda x: 1 if x > 0 else 0)
    correlations = df_corr.drop(['class', 'class_binary'], axis=1).corrwith(df_corr['class_binary']).abs().to_dict()
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'correlations': correlations
    }
    
    return model, scaler, metrics

def main():
    # Header Section
    st.markdown("""
        <div class="main-container">
            <h1 class="main-title">Cardiovascular Diseases Prediction Analysis</h1>
            <p class="subtitle">AI-Powered Cardiovascular Disease Risk Assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Theme Toggle
    col_space, col_toggle = st.columns([0.85, 0.15])
    with col_toggle:
        if st.session_state.theme == "light":
            if st.button("🌙 Dark Mode", use_container_width=True):
                st.session_state.theme = "dark"
                st.rerun()
        else:
            if st.button("☀️ Light Mode", use_container_width=True):
                st.session_state.theme = "light"
                st.rerun()
                
    # Load data and train model
    data = load_data()
    if data is None:
        return
    
    model, scaler, metrics = train_model(data)
    
    # Quick Stats (Evaluated dynamically)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(data)}</div>
                <div class="metric-label">Patients Analyzed</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{data.shape[1] - 1}</div>
                <div class="metric-label">Features</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['accuracy'] * 100:.1f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['recall'] * 100:.1f}%</div>
                <div class="metric-label">Sensitivity (Recall)</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create Tabs for Prediction, Model Performance, and Cohort Insights
    tab1, tab2, tab3 = st.tabs(["🔮 Patient Risk Predictor", "📊 AI Model Performance", "🔬 Clinical Population Insights"])
    
    with tab1:
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
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            st.markdown("### 👤 Personal Information")
            age = st.number_input("**Age** (years)", min_value=1, max_value=120, value=50, 
                                help="Patient's age in years", key="age_input")
            sex = st.selectbox("**Sex**", ["Female", "Male"], 
                              help="Patient's gender", key="sex_input")
            cp = st.selectbox("**Chest Pain Type**", 
                             ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                             help="Type of chest pain experienced", key="cp_input")
            
            st.markdown("### 💓 Cardiovascular Metrics")
            trestbps = st.number_input("**Resting Blood Pressure** (mm Hg)", min_value=0, max_value=300, value=120,
                                      help="Blood pressure at rest", key="trestbps_input")
            chol = st.number_input("**Serum Cholesterol** (mg/dl)", min_value=0, max_value=600, value=200,
                                  help="Cholesterol level in blood", key="chol_input")
            fbs = st.selectbox("**Fasting Blood Sugar > 120 mg/dl**", ["No", "Yes"],
                              help="Is fasting blood sugar elevated?", key="fbs_input")
            restecg = st.selectbox("**Resting ECG Results**", 
                                   ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                                   help="Electrocardiogram results at rest", key="restecg_input")
        
        with col_form2:
            st.markdown("### 🏃 Exercise & Test Results")
            thalach = st.number_input("**Maximum Heart Rate Achieved**", min_value=0, max_value=250, value=150,
                                     help="Peak heart rate during exercise", key="thalach_input")
            exang = st.selectbox("**Exercise Induced Angina**", ["No", "Yes"],
                                help="Does exercise cause chest pain?", key="exang_input")
            oldpeak = st.number_input("**ST Depression (Oldpeak)**", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                      help="ST segment depression during exercise", key="oldpeak_input")
            slope = st.selectbox("**Slope of Peak Exercise ST Segment**", 
                                ["Upsloping", "Flat", "Downsloping"],
                                help="Slope of ST segment during peak exercise", key="slope_input")
            ca = st.selectbox("**Number of Major Vessels** (0-3)", [0, 1, 2, 3],
                             help="Number of major vessels visible in fluoroscopy", key="ca_input")
            thal = st.selectbox("**Thalassemia Type**", 
                               ["Normal", "Fixed Defect", "Reversible Defect"],
                               help="Type of thalassemia", key="thal_input")
        
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
                input_scaled = scaler.transform(input_data)
                prediction = model.predict_proba(input_scaled)[0][1]
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

    with tab2:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 Neural Network Model Metrics</p>', unsafe_allow_html=True)
        
        st.write("""
        This prediction portal uses a Multilayer Perceptron (MLP) Neural Network classifier.
        Below are the validation metrics evaluated dynamically on a 20% validation split.
        """)
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric(label="Model Accuracy", value=f"{metrics['accuracy'] * 100:.1f}%", help="Percentage of overall correct diagnoses")
        with perf_col2:
            st.metric(label="Model Precision", value=f"{metrics['precision'] * 100:.1f}%", help="Percentage of flagged patients who actually have disease")
        with perf_col3:
            st.metric(label="Model Sensitivity (Recall)", value=f"{metrics['recall'] * 100:.1f}%", help="Percentage of actual disease cases successfully detected")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Grid layout for confusion matrix & risk factors
        grid_col1, grid_col2 = st.columns(2)
        with grid_col1:
            st.markdown("### 🧩 Confusion Matrix")
            st.markdown(f"""
                <div style="margin: 1rem 0; font-family: sans-serif; border: 1px solid var(--form-border); border-radius: 8px; overflow: hidden;">
                    <table style="width: 100%; border-collapse: collapse; text-align: center;">
                        <thead>
                            <tr style="background-color: var(--form-bg); border-bottom: 2px solid var(--form-border);">
                                <th style="padding: 12px; color: var(--text-color);">Actual \\ Predicted</th>
                                <th style="padding: 12px; color: var(--text-color); background-color: rgba(76, 175, 80, 0.1);">Predicted Healthy</th>
                                <th style="padding: 12px; color: var(--text-color); background-color: rgba(255, 107, 107, 0.1);">Predicted Disease</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid var(--form-border);">
                                <td style="padding: 12px; font-weight: bold; color: var(--text-color);">Actual Healthy</td>
                                <td style="padding: 12px; color: var(--text-color); font-weight: bold; background-color: rgba(76, 175, 80, 0.05);">{metrics['tn']} (True Neg)</td>
                                <td style="padding: 12px; color: var(--text-color); background-color: rgba(255, 107, 107, 0.05);">{metrics['fp']} (False Pos)</td>
                            </tr>
                            <tr>
                                <td style="padding: 12px; font-weight: bold; color: var(--text-color);">Actual Disease</td>
                                <td style="padding: 12px; color: var(--text-color); background-color: rgba(76, 175, 80, 0.05);">{metrics['fn']} (False Neg)</td>
                                <td style="padding: 12px; color: var(--text-color); font-weight: bold; background-color: rgba(255, 107, 107, 0.05);">{metrics['tp']} (True Pos)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            """, unsafe_allow_html=True)
            
        with grid_col2:
            st.markdown("### 📈 Top Clinical Risk Factors")
            corr_data = pd.DataFrame({
                'Feature': [
                    'Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Serum Cholesterol',
                    'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate',
                    'Exercise Induced Angina', 'ST Depression', 'ST Slope',
                    'Major Vessels', 'Thalassemia'
                ],
                'Correlation': [
                    metrics['correlations']['age'], metrics['correlations']['sex'], metrics['correlations']['cp'],
                    metrics['correlations']['trestbps'], metrics['correlations']['chol'], metrics['correlations']['fbs'],
                    metrics['correlations']['restecg'], metrics['correlations']['thalach'], metrics['correlations']['exang'],
                    metrics['correlations']['oldpeak'], metrics['correlations']['slope'], metrics['correlations']['ca'],
                    metrics['correlations']['thal']
                ]
            }).sort_values(by='Correlation', ascending=False)
            
            # Altair correlation bar chart
            corr_chart = alt.Chart(corr_data).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('Correlation:Q', title='Correlation Strength with Heart Disease'),
                y=alt.Y('Feature:N', sort='-x', title='Clinical Feature'),
                color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='purples'), legend=None)
            ).properties(
                height=250
            )
            st.altair_chart(corr_chart, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🔬 Clinical Cohort Comparison</p>', unsafe_allow_html=True)
        st.write("""
        Compare this patient's clinical parameters with the averages of the healthy cohort and the heart disease cohort.
        """)
        
        healthy = data[data['class'] == 0]
        diseased = data[data['class'] > 0]
        
        # Population Metrics Cards
        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
        with comp_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Age</div>
                    <div class="metric-value" style="font-size: 1.4rem; color: var(--accent-color) !important;">Patient: {age} yrs</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; color: var(--footer-text);">
                        Healthy Avg: {healthy['age'].mean():.1f}<br>
                        Disease Avg: {diseased['age'].mean():.1f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with comp_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Resting Blood Pressure</div>
                    <div class="metric-value" style="font-size: 1.4rem; color: var(--accent-color) !important;">Patient: {trestbps}</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; color: var(--footer-text);">
                        Healthy Avg: {healthy['trestbps'].mean():.1f}<br>
                        Disease Avg: {diseased['trestbps'].mean():.1f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with comp_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Serum Cholesterol</div>
                    <div class="metric-value" style="font-size: 1.4rem; color: var(--accent-color) !important;">Patient: {chol}</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; color: var(--footer-text);">
                        Healthy Avg: {healthy['chol'].mean():.1f}<br>
                        Disease Avg: {diseased['chol'].mean():.1f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with comp_col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Max Heart Rate</div>
                    <div class="metric-value" style="font-size: 1.4rem; color: var(--accent-color) !important;">Patient: {thalach}</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; color: var(--footer-text);">
                        Healthy Avg: {healthy['thalach'].mean():.1f}<br>
                        Disease Avg: {diseased['thalach'].mean():.1f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive selector and bar chart
        st.markdown("### 📊 Interactive Parameter Comparison")
        selected_feature = st.selectbox(
            "Select feature to visualize comparison:", 
            ["Age (years)", "Resting Blood Pressure (mm Hg)", "Serum Cholesterol (mg/dl)", "Max Heart Rate Achieved (bpm)"],
            key="feature_comparison_select"
        )
        
        feature_map = {
            "Age (years)": ("age", age, "Age"),
            "Resting Blood Pressure (mm Hg)": ("trestbps", trestbps, "Blood Pressure"),
            "Serum Cholesterol (mg/dl)": ("chol", chol, "Cholesterol"),
            "Max Heart Rate Achieved (bpm)": ("thalach", thalach, "Max Heart Rate")
        }
        
        db_col, pat_val, label = feature_map[selected_feature]
        healthy_avg = healthy[db_col].mean()
        disease_avg = diseased[db_col].mean()
        
        comp_df = pd.DataFrame({
            'Cohort': ['Patient', 'Healthy Cohort', 'Heart Disease Cohort'],
            'Value': [pat_val, healthy_avg, disease_avg]
        })
        
        compare_chart = alt.Chart(comp_df).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X('Cohort:N', sort=None, title=None),
            y=alt.Y('Value:Q', title=selected_feature),
            color=alt.Color('Cohort:N', scale=alt.Scale(
                domain=['Patient', 'Healthy Cohort', 'Heart Disease Cohort'],
                range=['#667eea', '#4CAF50', '#FF6B6B']
            ), legend=None)
        ).properties(
            height=300
        )
        st.altair_chart(compare_chart, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

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
