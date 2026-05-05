# House Price Prediction ML System

## 🎯 Business Problem
Predict house prices based on location, size, age, and features for real estate valuation.

## 📊 Dataset
- `data/raw/house_prices.csv`: 300 samples
- Features: Area, Bedrooms, Bathrooms, Age, Location, Property_Type
- Target: Price (₹)

## 🚀 Quick Start
```bash
# Activate venv (create if needed: python -m venv venv)
# Windows: venv\\Scripts\\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt

# EDA
jupyter notebook notebooks/01_eda.ipynb

# Train model
python src/model_training.py

# Launch web app
streamlit run app/web_app.py
```

## 🏗️ Structure
```
├── data/raw/          # Raw data
├── data/processed/    # Processed data
├── notebooks/         # EDA & experiments
├── src/               # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_inference.py
├── app/               # Web app
│   └── web_app.py
├── models/            # Saved models
├── tests/             # Tests
├── config/            # Configs
├── TODO.md            # Progress
└── README.md
```

## 📈 Models Compared
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

**Expected Performance:** R² ~0.85, MAE ~₹400k

## 🔮 Web App
Input features → Predict price + confidence + feature insights.

## 📝 Usage
1. Train: Generates `models/best_model.joblib` + report
2. Predict via app or `python src/model_inference.py`
3. API-ready modular design.

## 🧪 Tests
```bash
pytest tests/
```

## 📄 Documentation
See `docs/report.md` for metrics, feature importance, insights.

