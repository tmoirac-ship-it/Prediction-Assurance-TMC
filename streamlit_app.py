import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

st.set_page_config(page_title="Segmentation Assurance", page_icon="🚗", layout="wide")

MODEL_FILE = "model.pkl"
DATASET_FILE = "merged_dataset.csv"

# ==============================================================================
# ENTRAINEMENT
# ==============================================================================

def train_model():
    df = pd.read_csv(DATASET_FILE)
    df = df[df['dataset_type'] == 'train'].copy()
    df = df.drop(['id', 'dataset_type'], axis=1)

    # Encodage
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Vehicle_Age'] = le.fit_transform(df['Vehicle_Age'])
    df['Vehicle_Damage'] = le.fit_transform(df['Vehicle_Damage'])

    df = df.dropna()

    X = df.drop('Response', axis=1)
    y = df['Response']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    }

    model_data = {
        "model": model,
        "scaler": scaler,
        "features": list(X.columns),
        "metrics": metrics
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)

    return model_data

# ==============================================================================
# CHARGEMENT
# ==============================================================================

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return train_model()

model_data = load_model()
model = model_data["model"]
scaler = model_data["scaler"]
metrics = model_data["metrics"]

# ==============================================================================
# INTERFACE
# ==============================================================================

st.title("🚗 Segmentation de Marché - Assurance")

st.sidebar.header("📊 Performances")
for k, v in metrics.items():
    st.sidebar.metric(k, f"{v:.2%}")

st.header("📝 Prédiction Client")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Genre", ["Male", "Female"])
    age = st.slider("Age", 18, 85, 30)
    driving_license = st.selectbox("Permis", [0, 1])
    region_code = st.number_input("Code Région", 1, 100, 28)
    previously_insured = st.selectbox("Déjà assuré", [0, 1])

with col2:
    vehicle_age = st.selectbox("Age véhicule", ["< 1 Year", "1-2 Year", "> 2 Years"])
    vehicle_damage = st.selectbox("Dommage", ["Yes", "No"])
    annual_premium = st.number_input("Prime annuelle", 1000, 200000, 30000)
    policy_sales_channel = st.number_input("Canal vente", 1, 200, 152)
    vintage = st.slider("Ancienneté client", 10, 300, 200)

if st.button("🔮 Prédire"):

    gender_encoded = 1 if gender == "Male" else 0
    vehicle_age_encoded = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}[vehicle_age]
    vehicle_damage_encoded = 1 if vehicle_damage == "Yes" else 0

    input_data = np.array([[gender_encoded, age, driving_license,
                            region_code, previously_insured,
                            vehicle_age_encoded, vehicle_damage_encoded,
                            annual_premium, policy_sales_channel, vintage]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.subheader("Résultat")

    if prediction == 1:
        st.success("Client probable répondant ✅")
    else:
        st.error("Client probable non-répondant ❌")

    fig, ax = plt.subplots()
    ax.barh(["Non", "Oui"], proba)
    ax.set_xlim(0, 1)
    st.pyplot(fig)
