# 🚗 Segmentation de Marché dans l'Assurance - Application Streamlit

Une application web de Machine Learning pour prédire si un client va répondre positivement à une offre d'assurance véhicule.

**L'application entraîne automatiquement le modèle au premier lancement!**

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red)
![scikit-learn](https://img.shields.io/badge/scikit-learn-1.2+-green)

## 📋 Description

Cette application utilise des algorithmes de Machine Learning pour:
- Prédire la réponse d'un client à une offre d'assurance véhicule
- Analyser les facteurs qui influencent la décision du client
- Visualiser les performances du modèle

L'application **entraîne automatiquement le modèle** lors du premier lancement si celui-ci n'existe pas.

## 🏗️ Architecture du Projet

```
├── streamlit_app.py      # Application principale Streamlit (contient tout le code)
├── requirements.txt     # Dépendances Python
├── merged_dataset.csv    # Dataset d'assurance
├── auto-mpg.pkl          # Modèle entraîné (généré automatiquement)
└── README.md            # Ce fichier
```

## ⚡ Installation et Lancement

### 1. Cloner le dépôt
```
bash
git clone <url-du-depot>
cd <nom-du-projet>
```

### 2. Créer un environnement virtuel (conseillé)
```
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```
bash
pip install -r requirements.txt
```

### 4. Lancer l'application
```bash
streamlit run streamlit_app.py
```

**L'application va:**
- Détecter si le modèle existe
- Si nécessaire, entraîner automatiquement le modèle (environ 30 secondes)
- Lancer l'interface web

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse: `http://localhost:8501`

## ☁️ Déploiement sur Streamlit Cloud

### Prérequis
1. Un compte [Streamlit Cloud](https://streamlit.io/cloud)
2. Un dépôt GitHub contenant le projet

### Étapes de déploiement

1. **Pousser le code sur GitHub**
   - Assurez-vous que `auto-mpg.pkl` est inclus dans votre dépôt
   - Le fichier doit être généré localement avant le push

2. **Connecter Streamlit Cloud à GitHub**
   - Allez sur [Streamlit Cloud](https://streamlit.io/cloud)
   - Connectez votre compte GitHub
   - Sélectionnez votre dépôt

3. **Configurer le déploiement**
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Python version: 3.8 ou supérieur

4. **Déployer**
   - Cliquez sur "Deploy!"
   - L'application sera automatiquement déployée

## 📊 Fonctionnalités

### Page Principale
- **Formulaire de prédiction**: Saisie des informations client et véhicule
- **Prédiction en temps réel**: Résultat instantané avec probabilités
- **Visualisation**: Graphique des probabilités de réponse

### Sidebar
- Informations sur le modèle utilisé
- Métriques de performance (Accuracy, Precision, Recall, F1, ROC-AUC)

### Onglet Analyse
- Description du dataset
- Importance des features
- Explication des métriques

## 🤖 Modèles Utilisés

Le projet compare plusieurs algorithmes:
- KNN (K-Nearest Neighbors)
- Régression Logistique
- Arbre de Décision
- Random Forest
- Gradient Boosting
- Naive Bayes

Le modèle final utilise **Gradient Boosting** pour sa performance supérieure.

## 📈 Métriques de Performance

| Métrique | Description |
|----------|-------------|
| Accuracy | Proportion de prédictions correctes |
| Precision | Proportion de répondants identifiés qui sont vraiment des répondants |
| Recall | Proportion de répondants réels qui sont identifiés |
| F1 Score | Moyenne harmonique de Precision et Recall |
| ROC-AUC | Mesure de la capacité du modèle à distinguer les classes |

## ⚠️ Dépannage

### Erreur: "Modèle non trouvé"
- Exécutez d'abord `python train_model.py` pour générer le fichier `auto-mpg.pkl`

### Erreur: "ModuleNotFoundError"
- Réinstallez les dépendances: `pip install -r requirements.txt`

### Erreur sur Streamlit Cloud
- Vérifiez que `auto-mpg.pkl` est présent dans votre dépôt Git
- Le fichier doit être généré localement et commité

## 📝 Licence

Ce projet est à des fins éducatives.

## 👤 Auteur

Projet ML - Segmentation de Marché dans l'Assurance
