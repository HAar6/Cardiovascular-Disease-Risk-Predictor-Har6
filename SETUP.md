# Setup Instructions

## Quick Start

1. **Install Python 3.7+** (if not already installed)

2. **Navigate to the project folder:**
   ```bash
   cd Cardiovascular-Disease-Predictor
   ```

3. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Mac/Linux:
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

## Troubleshooting

- **Port already in use?** Streamlit will automatically use the next available port (8502, 8503, etc.)
- **Import errors?** Make sure all dependencies are installed: `pip install -r requirements.txt`
- **Model training slow?** This is normal on first run. The model is cached for subsequent runs.

## For GitHub Upload

1. Initialize git repository:
   ```bash
   git init
   ```

2. Add all files:
   ```bash
   git add .
   ```

3. Commit:
   ```bash
   git commit -m "Initial commit: Cardiovascular Disease Predictor"
   ```

4. Create a new repository on GitHub and push:
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```

