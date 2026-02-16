# Projet CYB6033 – Détection d’Intrusions avec l’IA

## Description du projet

Ce projet consiste à développer et évaluer des modèles d’apprentissage supervisé pour la **détection d’intrusions réseau** à partir du dataset NSL-KDD.  
L’objectif est de comparer plusieurs algorithmes classiques de machine learning pour identifier les attaques informatiques et distinguer les connexions normales des connexions malveillantes.  

Le projet a été réalisé dans le cadre du cours **CYB6033 – Méthodes avancées en cybersécurité basées sur l’intelligence artificielle**.

---

## 1. Préparation du dataset

Les étapes suivantes ont été appliquées pour préparer les données :

1. **Nettoyage des données**  
   - Suppression des valeurs manquantes ou incohérentes.  
   - Vérification des doublons et suppression si nécessaire.

2. **Encodage des variables catégorielles**  
   - Les colonnes `protocol_type`, `service` et `flag` ont été encodées avec **One-Hot Encoding** pour être utilisées par les modèles supervisés.

3. **Normalisation / standardisation des features**  
   - Les variables numériques ont été standardisées pour garantir une contribution équitable de chaque feature.

4. **Séparation des données**  
   - Le dataset a été divisé en **jeu d’entraînement** (`X_train`, `y_train`) et **jeu de test** (`X_test`, `y_test`) avec un ratio 80/20.

---

## 2. Algorithmes appliqués

Les modèles supervisés suivants ont été implémentés et testés :

### a) Régression Logistique
- **Principe** : Modèle linéaire pour la classification binaire, utilisant la fonction sigmoïde pour prédire la probabilité d’appartenance à une classe.  
- **Paramètres** : `max_iter=500` pour assurer la convergence.  
- **Évaluation** : précision, rappel, F1-score, courbe ROC.

### b) Arbre de Décision
- **Principe** : Divise les données en fonction des features les plus discriminantes pour créer un arbre de classification.  
- **Paramètres** : `max_depth=10` pour limiter la complexité.  
- **Évaluation** : précision, rappel, F1-score, matrice de confusion.

### c) Forêt Aléatoire
- **Principe** : Ensemble d’arbres de décision combinés par **bagging** pour réduire la variance et améliorer la généralisation.  
- **Paramètres** : `n_estimators=100`, `max_depth=10`, `random_state=42`.  
- **Évaluation** : mêmes métriques que précédemment.

### d) Naive Bayes
- **Principe** : Classificateur probabiliste basé sur le théorème de Bayes, supposant l’indépendance conditionnelle entre les features.  
- **Paramètres** : valeurs par défaut (`GaussianNB()`).  
- **Évaluation** : précision, rappel, F1-score, AUC.

---

## 3. Évaluation des modèles

Pour chaque modèle, les métriques suivantes ont été calculées et stockées dans `results/metrics.txt` :

- **Accuracy (Exactitude)** : proportion de prédictions correctes.  
- **Precision (Précision)** : proportion des prédictions positives correctes.  
- **Recall (Rappel)** : capacité à détecter toutes les attaques réelles.  
- **F1-Score** : moyenne harmonique entre précision et rappel.  
- **ROC & AUC** : performance globale pour discriminer les classes.  
- **Matrice de confusion** : visualisation des prédictions correctes et des erreurs, sauvegardée sous forme d’image.

---

## 4. Comparaison et analyse des résultats

Après l’évaluation :

- **Forêt Aléatoire** : Meilleure performance globale avec précision élevée et AUC proche de 1.  
- **Arbre de Décision** : Performances correctes mais légèrement inférieures, sensible au surapprentissage.  
- **Régression Logistique** : Bonnes performances sur les classes binaires, moins efficace pour les attaques rares.  
- **Naive Bayes** : Simple et rapide, mais moins précis que les modèles basés sur les arbres.

**Conclusion** : La **forêt aléatoire** est le modèle le plus performant pour ce dataset, combinant robustesse et capacité à gérer les features numériques et catégorielles.

---

## 5. Améliorations potentielles

- Ajustement des **hyperparamètres** via **GridSearchCV** ou **RandomSearch**.  
- Tester des algorithmes avancés comme **Gradient Boosting**, **XGBoost**, ou **SVM**.  
- Analyse des features importantes pour identifier les caractéristiques critiques pour la détection d’intrusions.  
- Extension de l’étude à des datasets plus volumineux et récents pour valider la généralisation.

---

## 6. Structure du projet

project/
│
├── data/
│ └── KDDTrain.csv
│
├── results/
│ ├── metrics.txt
│ ├── ConfusionMatrix_Logistic Regression.png
│ ├── ConfusionMatrix_Decision Tree.png
│ ├── ConfusionMatrix_Random Forest.png
│ └── roc_curve.png
│
├── utils.py # fonctions evaluate_model, plot_confusion_matrix, save_metrics
├── data_preparation.py # préparation et séparation des données
├── trainmodel.py # script principal d’entraînement et d’évaluation
└── README.md # ce fichier



---

## 7. Instructions d’exécution

1. Installer les dépendances :

```bash
pip install -r requirements.txt


## Lancer le script principal
python train_model.py

