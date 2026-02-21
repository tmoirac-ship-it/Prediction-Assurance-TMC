"""
================================================================================
APPLICATION STREAMLIT: Segmentation de Marché dans l'Assurance
================================================================================
Cette application web permet de prédire si un client va répondre
positivement à une offre d'assurance véhicule.

Pour lancer l'application:
    streamlit run streamlit_app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, auc)

# ================================================================================
# CONFIGURATION DE LA PAGE
# ================================================================================

st.set_page_config(
    page_title="Segmentation Assurance",
    page_icon="🚗",
    layout="wide"
)

# ================================================================================
# CHARGEMENT DU MODÈLE
# ================================================================================

@st.cache_resource
def load_model():
    with open('auto-mpg.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    best_model_name = model_data['best_model_name']
    metrics = model_data['metrics']
except:
    st.error("❌ Modèle non trouvé. Veuillez d'abord exécuter projetIA.py")
    st.stop()

# ================================================================================
# EN-TÊTE
# ================================================================================

st.title("🚗 Segmentation de Marché dans l'Assurance")
st.markdown("""
Cette application utilise le **Machine Learning** pour prédire si un client
va répondre positivement à une offre d'assurance véhicule.
""")

# ================================================================================
# SIDEBAR - INFORMATIONS
# ================================================================================

st.sidebar.header("ℹ️ Informations")
st.sidebar.write(f"**Modèle utilisé:** {best_model_name}")
st.sidebar.write(f"**Features:** {len(features)}")
st.sidebar.markdown("---")
st.sidebar.header("📊 Performances du modèle")
st.sidebar.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
st.sidebar.metric("Precision", f"{metrics['Precision']:.1%}")
st.sidebar.metric("Recall", f"{metrics['Recall']:.1%}")
st.sidebar.metric("F1 Score", f"{metrics['F1']:.1%}")
st.sidebar.metric("ROC-AUC", f"{metrics['ROC-AUC']:.1%}")

# ================================================================================
# FORMULAIRE DE PRÉDICTION
# ================================================================================

st.header("📝 Prédiction de réponse client")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Informations client")
    
    gender = st.selectbox("Genre", ["Male", "Female"], help="Genre du client")
    age = st.slider("Âge", 18, 85, 30, help="Âge du client en années")
    
    driving_license = st.selectbox("Permis de conduire", [0, 1], 
                                   format_func=lambda x: "Oui" if x == 1 else "Non")
    
    region_code = st.number_input("Code région", min_value=1, max_value=100, value=28)
    
    previously_insured = st.selectbox("Assurance véhicule préalable", [0, 1],
                                     format_func=lambda x: "Oui" if x == 1 else "Non")

with col2:
    st.subheader("Informations véhicule")
    
    vehicle_age = st.selectbox("Âge du véhicule", 
                               ["< 1 Year", "1-2 Year", "> 2 Years"],
                               help="Ancienneté du véhicule")
    
    vehicle_damage = st.selectbox("Dommage véhicule précédent", ["Yes", "No"],
                                  help="Le véhicule a-t-il eu des dommages?")
    
    annual_premium = st.number_input("Prime annuelle ($)", 
                                     min_value=1000, max_value=200000, value=30000)
    
    policy_sales_channel = st.number_input("Canal de vente", 
                                           min_value=1, max_value=200, value=152)
    
    vintage = st.slider("Nombre de jours client", 10, 300, 200)

# ================================================================================
# PRÉDICTION
# ================================================================================

# Encoder les valeurs
gender_encoded = 1 if gender == "Male" else 0
vehicle_age_encoded = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}[vehicle_age]
vehicle_damage_encoded = 1 if vehicle_damage == "Yes" else 0

# Créer le vecteur de features - ORDRE CORRECT
input_data = np.array([[gender_encoded, age, driving_license, region_code, 
                        previously_insured, vehicle_age_encoded, vehicle_damage_encoded,
                        annual_premium, policy_sales_channel, vintage]])

# Afficher les features pour débogage
st.write("Features attendue:", features)
st.write("Input shape:", input_data.shape)

# Normaliser les données
input_scaled = scaler.transform(input_data)

# Bouton de prédiction
if st.button("🔮 Prédire", type="primary"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    st.subheader("Résultat de la prédiction")
    
    if prediction == 1:
        st.success("✅ Le client va probablement RÉPONDRE à l'offre d'assurance!")
    else:
        st.warning("❌ Le client va probablement NE PAS RÉPONDRE à l'offre d'assurance!")
    
    # Afficher les probabilités
    col_proba1, col_proba2 = st.columns(2)
    with col_proba1:
        st.metric("Probabilité Non-répondant", f"{probability[0]:.1%}")
    with col_proba2:
        st.metric("Probabilité Répondant", f"{probability[1]:.1%}")
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(["Non-répondant", "Répondant"], probability, 
                   color=['coral', 'steelblue'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilité')
    ax.set_title('Probabilité de réponse')
    for bar, prob in zip(bars, probability):
        ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{prob:.1%}', va='center')
    st.pyplot(fig)

# ================================================================================
# ANALYSE DES DONNÉES
# ================================================================================

st.markdown("---")
st.header("📊 Analyse des données")

tab1, tab2, tab3 = st.tabs(["Description", "Features", "Métriques"])

with tab1:
    st.subheader("À propos du dataset")
    st.write("""
    Ce dataset provient d'un challenge organisé par une compagnie d'assurance américaine.
    Il contient des informations sur les clients et leurs véhicules.
    
    **Objectif:** Prédire si un client va responder positivement (Response=1)
    à une offre d'assurance véhicule.
    
    **Features utilisés:**
    - Gender: Genre du client
    - Age: Âge du client
    - Driving_License: Permis de conduire (1=Oui, 0=Non)
    - Region_Code: Code de la région
    - Previously_Insured: Assurance véhicule préalable
    - Vehicle_Age: Âge du véhicule
    - Vehicle_Damage: Dommage véhicule précédent
    - Annual_Premium: Prime annuelle
    - Policy_Sales_Channel: Canal de vente
    - Vintage: Nombre de jours depuis que le client est dans la base
    """)

with tab2:
    st.subheader("Importance des features")
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['Feature'], importance['Importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title('Importance des features - ' + best_model_name)
        st.pyplot(fig)
    else:
        st.info("L'importance des features n'est pas disponible pour ce modèle.")

with tab3:
    st.subheader("Métriques de performance")
    
    metric_df = pd.DataFrame({
        'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Valeur': [metrics['Accuracy'], metrics['Precision'], 
                   metrics['Recall'], metrics['F1'], metrics['ROC-AUC']]
    })
    
    st.table(metric_df.set_index('Métrique'))
    
    st.info("""
    **Explication des métriques:**
    - **Accuracy:** Proportion de prédictions correctes
    - **Precision:** Proportion de répondants identifiés qui sont vraiment des répondants
    - **Recall:** Proportion de répondants réels qui sont identifiés
    - **F1 Score:** Moyenne harmonique de Precision et Recall
    - **ROC-AUC:** Mesure de la capacité du modèle à distinguer les classes
    """)

# ================================================================================
# PIED DE PAGE
# ================================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Projet ML - Segmentation de Marché dans l'Assurance</p>
    <p>Créé avec Streamlit et scikit-learn</p>
</div>
""", unsafe_allow_html=True)
