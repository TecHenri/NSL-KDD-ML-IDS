import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from data_preparation import load_and_prepare_data
from utils import evaluate_model, plot_confusion_matrix, save_metrics_all


def main():

    # Chemin dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "KDDTrain.csv")

    # Préparation des données
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_path)

    # Définition modèles
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Naive Bayes": GaussianNB()
    }

    plt.figure(figsize=(8,6))
    all_metrics = {}  # dictionnaire pour stocker toutes les métriques

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics_dict  = evaluate_model(model, X_test, y_test, name)
        all_metrics[name] = metrics_dict  # stocker les métriques dans le dictionnaire
        # Confusion matri
        plot_confusion_matrix(metrics_dict)
        save_metrics_all(all_metrics)  # sauvegarder toutes les métriques dans un seul fichier
""" 
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics_dict  = evaluate_model(model, X_test, y_test, name)
        3#all_metrics[name] = metrics_dict  # stocker les métriques dans le dictionnaire
        # Courbe ROC
        fpr, tpr, _ = metrics_dict["roc_curve"]
        auc_score = metrics_dict["roc_auc"]
        plt.plot(fpr, tpr, label=f"{metrics_dict['model_name']} (AUC={auc_score:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbe ROC - Comparaison des modèles")
    plt.legend()

    results_path = os.path.join(BASE_DIR, "results", "roc_curve.png")
    plt.savefig(results_path)
    plt.show() """


if __name__ == "__main__":
    main()
    print("Entraînement et évaluation terminés. Résultats sauvegardés dans le dossier 'results'.")
