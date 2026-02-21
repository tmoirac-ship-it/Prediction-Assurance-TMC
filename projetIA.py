"""
================================================================================
PROJET ML: Segmentation de Marché dans l'Assurance
================================================================================
Ce script implémente un modèle de prédiction de la réponse des clients
à une offre d'assurance automobile.

Objectif: Prédire si un client va répondre positivement (Response=1)
à une offre d'assurance véhicule.

Bibliothèques requises:
- pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit

Installation: pip install -r requirements.txt
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# ================================================================================

print("=" * 80)
print("1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES")
print("=" * 80)

# Chargement du dataset d'assurance
df = pd.read_csv('merged_dataset.csv')

print(f"✓ Dataset chargé avec succès")
print(f"\n📊 Shape du dataset: {df.shape}")

# Filtrer seulement les données d'entraînement
df = df[df['dataset_type'] == 'train'].copy()
print(f"✓ Après filtrage (train only): {df.shape[0]} lignes")

# Supprimer les colonnes non nécessaires
df = df.drop(['id', 'dataset_type'], axis=1)

print(f"\n📋 Colonnes disponibles: {list(df.columns)}")
print(f"\n📋 Types de données:\n{df.dtypes}")

# ================================================================================
# 2. ANALYSE EXPLORATOIRE DES DONNÉES
# ================================================================================

print("\n" + "=" * 80)
print("2. ANALYSE EXPLORATOIRE DES DONNÉES")
print("=" * 80)

# Valeurs manquantes
print("\n📋 Valeurs manquantes:")
print(df.isnull().sum())

# Statistiques descriptives
print("\n📈 Statistiques descriptives (numériques):")
print(df.describe())

# Distribution de la cible
print("\n🎯 Distribution de la cible (Response):")
print(df['Response'].value_counts())
print(f"\nTaux de réponse: {df['Response'].mean()*100:.2f}%")

# ================================================================================
# 3. PRÉTRAITEMENT DES DONNÉES
# ================================================================================

print("\n" + "=" * 80)
print("3. PRÉTRAITEMENT DES DONNÉES")
print("=" * 80)

# Encodage des variables catégorielles
le_gender = LabelEncoder()
le_vehicle_age = LabelEncoder()
le_vehicle_damage = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Vehicle_Age'] = le_vehicle_age.fit_transform(df['Vehicle_Age'])
df['Vehicle_Damage'] = le_vehicle_damage.fit_transform(df['Vehicle_Damage'])

print(f"✓ Types après encodage:")
print(df.dtypes)

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()
print(f"\n✓ Après suppression des NA: {df.shape[0]} lignes")

# ================================================================================
# 4. CRÉATION DES VISUALISATIONS
# ================================================================================

print("\n" + "=" * 80)
print("4. CRÉATION DES VISUALISATIONS")
print("=" * 80)

# Visualisation EDA
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution de l'âge
axes[0, 0].hist(df['Age'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution de l\'âge')
axes[0, 0].set_xlabel('Âge')
axes[0, 0].set_ylabel('Fréquence')

# Distribution par genre
gender_counts = df['Gender'].value_counts()
axes[0, 1].pie(gender_counts, labels=['Homme', 'Femme'], autopct='%1.1f%%', colors=['steelblue', 'coral'])
axes[0, 1].set_title('Distribution par genre')

# Distribution par réponse
response_counts = df['Response'].value_counts()
axes[1, 0].pie(response_counts, labels=['Non répondant', 'Répondant'], autopct='%1.1f%%', colors=['lightgray', 'green'])
axes[1, 0].set_title('Distribution de la réponse')

# Age vs Response
df.boxplot(column='Age', by='Response', ax=axes[1, 1])
axes[1, 1].set_title('Âge par réponse')
axes[1, 1].set_xlabel('Response')
axes[1, 1].set_ylabel('Âge')

plt.tight_layout()
plt.savefig('eda_insurance.png', dpi=100)
print(f"✓ Visualisations sauvegardées dans 'eda_insurance.png'")

# Matrice de corrélation
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=100)
print(f"✓ Matrice de corrélation sauvegardée dans 'correlation_matrix.png'")

# ================================================================================
# 5. PRÉPARATION DES DONNÉES POUR LE MODÈLE
# ================================================================================

print("\n" + "=" * 80)
print("5. PRÉPARATION DES DONNÉES POUR LE MODÈLE")
print("=" * 80)

# Features et target
X = df.drop('Response', axis=1)
y = df['Response']

print(f"✓ Features: {list(X.columns)}")
print(f"✓ Target: Response (0 ou 1)")

# Échantillonner pour accélérer l'entraînement
if len(X) > 20000:
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=20000, stratify=y, random_state=42)
else:
    X_sample, y_sample = X, y

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)

print(f"✓ Train set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Données normalisées avec StandardScaler")

# ================================================================================
# 6. KNN - INFLUENCE DU PARAMÈTRE K
# ================================================================================

print("\n" + "=" * 80)
print("6. KNN - INFLUENCE DU PARAMÈTRE K")
print("=" * 80)

print("Recherche de la meilleure valeur de k...")

k_values = [3, 5, 7, 9, 11, 15, 20]
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, knn.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f"  k={k}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

# Meilleur k
best_k_idx = np.argmax(test_accuracies)
best_k = k_values[best_k_idx]

print(f"\n✓ Meilleur k: {best_k} (Accuracy: {test_accuracies[best_k_idx]:.3f})")

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, 'b-o', label='Train Accuracy')
plt.plot(k_values, test_accuracies, 'r-o', label='Test Accuracy')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Influence du paramètre k sur les performances KNN')
plt.legend()
plt.grid(True)
plt.savefig('knn_k_influence.png', dpi=100)
print(f"✓ Graphique sauvegardé dans 'knn_k_influence.png'")

# ================================================================================
# 7. COMPARAISON AVEC ET SANS NORMALISATION
# ================================================================================

print("\n" + "=" * 80)
print("7. COMPARAISON AVEC ET SANS NORMALISATION")
print("=" * 80)

# KNN sans normalisation
knn_no_scale = KNeighborsClassifier(n_neighbors=best_k)
knn_no_scale.fit(X_train, y_train)
y_pred_no_scale = knn_no_scale.predict(X_test)

# KNN avec normalisation
knn_with_scale = KNeighborsClassifier(n_neighbors=best_k)
knn_with_scale.fit(X_train_scaled, y_train)
y_pred_with_scale = knn_with_scale.predict(X_test_scaled)

print(f"✓ Accuracy sans normalisation: {accuracy_score(y_test, y_pred_no_scale):.3f}")
print(f"✓ Accuracy avec normalisation: {accuracy_score(y_test, y_pred_with_scale):.3f}")

# ================================================================================
# 8. ENTRAÎNEMENT DE PLUSIEURS MODÈLES
# ================================================================================

print("\n" + "=" * 80)
print("8. COMPARAISON DES MODÈLES")
print("=" * 80)

# Définir les modèles
models = {
    'KNN': KNeighborsClassifier(n_neighbors=best_k),
    'Régression Logistique': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Naive Bayes': GaussianNB()
}

results = []

for name, model in models.items():
    print(f"Entraînement {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = 0
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC-AUC': roc_auc
    })
    print(f"  -> Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")

# Créer le DataFrame des résultats
results_df = pd.DataFrame(results)

# ================================================================================
# 9. COMPARAISON DES PERFORMANCES DES MODÈLES
# ================================================================================

print("\n" + "=" * 80)
print("9. COMPARAISON DES PERFORMANCES DES MODÈLES")
print("=" * 80)

print(results_df.to_string(index=False))

# Trouver le meilleur modèle (par F1)
best_idx = results_df['F1'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_metrics = results_df.loc[best_idx].to_dict()

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
print(f"   Accuracy: {best_metrics['Accuracy']:.3f}")
print(f"   Precision: {best_metrics['Precision']:.3f}")
print(f"   Recall: {best_metrics['Recall']:.3f}")
print(f"   F1 Score: {best_metrics['F1']:.3f}")
print(f"   ROC-AUC: {best_metrics['ROC-AUC']:.3f}")

# Visualisation comparatif
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.15

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i*width, results_df[metric], width, label=metric)

ax.set_xlabel('Modèle')
ax.set_ylabel('Score')
ax.set_title('Comparaison des modèles')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig('models_comparison.png', dpi=100)
print(f"✓ Graphique comparatif sauvegardé dans 'models_comparison.png'")

# ================================================================================
# 10. MATRICE DE CONFUSION ET COURBE ROC
# ================================================================================

print("\n" + "=" * 80)
print("10. MATRICE DE CONFUSION ET COURBE ROC")
print("=" * 80)

# Matrice de confusion pour le meilleur modèle
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice de confusion - {best_model_name}')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
print(f"✓ Matrice de confusion sauvegardée dans 'confusion_matrix.png'")

# Courbes ROC
plt.figure(figsize=(10, 8))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=100)
print(f"✓ Courbes ROC sauvegardées dans 'roc_curves.png'")

# ================================================================================
# 11. SAUVEGARDE DU MODÈLE
# ================================================================================

print("\n" + "=" * 80)
print("11. SAUVEGARDE DU MODÈLE")
print("=" * 80)

# Entraîner le modèle final sur toutes les données échantillonnées
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X_sample)

# Utiliser Gradient Boosting comme modèle final
final_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
final_model.fit(X_scaled_full, y_sample)

# Sauvegarder le modèle
model_data = {
    'model': final_model,
    'scaler': scaler_full,
    'features': list(X.columns),
    'best_model_name': best_model_name,
    'metrics': {
        'Accuracy': float(best_metrics['Accuracy']),
        'Precision': float(best_metrics['Precision']),
        'Recall': float(best_metrics['Recall']),
        'F1': float(best_metrics['F1']),
        'ROC-AUC': float(best_metrics['ROC-AUC'])
    }
}

with open('auto-mpg.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"✓ Modèle sauvegardé dans 'auto-mpg.pkl'")

# ================================================================================
# 12. RÉSUMÉ FINAL
# ================================================================================

print("\n" + "=" * 80)
print("RÉSUMÉ FINAL - SEGMENTATION DE MARCHÉ DANS L'ASSURANCE")
print("=" * 80)

print(f"""
📊 PROJET: Segmentation de Marché dans l'Assurance
{'='*60}

1. DONNÉES:
   - Dataset: Insurance Response
   - {X_sample.shape[0]} échantillons, {X.shape[1]} features
   - Target: Response (0 = Non répondant, 1 = Répondant)
   - Taux de réponse: {df['Response'].mean()*100:.2f}%

2. KNN - Influence du paramètre k:
   - Meilleur k: {best_k}
   - Accuracy avec k={best_k}: {test_accuracies[best_k_idx]:.3f}

3. Effet de la normalisation:
   - La normalisation {'améliore' if accuracy_score(y_test, y_pred_with_scale) > accuracy_score(y_test, y_pred_no_scale) else 'n\'améliore pas'} les performances

4. Meilleurs modèles:
{results_df.to_string(index=False)}

5. 🏆 MEILLEUR MODÈLE: {best_model_name}
   - Accuracy: {best_metrics['Accuracy']:.3f}
   - Precision: {best_metrics['Precision']:.3f}
   - Recall: {best_metrics['Recall']:.3f}
   - F1 Score: {best_metrics['F1']:.3f}
   - ROC-AUC: {best_metrics['ROC-AUC']:.3f}

6. Fichiers générés:
   - auto-mpg.pkl: Modèle entraîné
   - eda_insurance.png: Analyse exploratoire
   - correlation_matrix.png: Matrice de corrélation
   - knn_k_influence.png: Influence de k
   - models_comparison.png: Comparaison des modèles
   - roc_curves.png: Courbes ROC
   - confusion_matrix.png: Matrice de confusion

✅ Projet terminé avec succès!
""")

print("=" * 80)
print("Fin du programme")
print("=" * 80)
