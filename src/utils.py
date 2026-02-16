import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Répertoire pour sauvegarder les résultats
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Évalue un modèle et retourne un dictionnaire contenant toutes les métriques et matrices.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": roc_curve(y_test, y_prob)  # retourne fpr, tpr, thresholds
    }

    return metrics


def plot_confusion_matrix(metrics_dict):
    """
    Affiche et sauvegarde la confusion matrix à partir du dictionnaire d'évaluation.
    """
    conf_matrix = metrics_dict["confusion_matrix"]
    model_name = metrics_dict["model_name"]

    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='0.0f')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    cm_path = os.path.join(results_dir, f"{model_name}_ConfusionMatrix.png")
    plt.savefig(cm_path)
    plt.show()


def plot_roc_curve(metrics_dict):
    """
    Affiche et sauvegarde la ROC curve à partir du dictionnaire d'évaluation.
    """
    fpr, tpr, _ = metrics_dict["roc_curve"]
    roc_auc = metrics_dict["roc_auc"]
    model_name = metrics_dict["model_name"]

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    roc_path = os.path.join(results_dir, f"{model_name}_ROC.png")
    plt.savefig(roc_path)
    plt.show()


def save_metrics_all(models_metrics, file_name="metrics.txt"):
    """
    Sauvegarde les métriques de tous les modèles dans un seul fichier.
    
    :param models_metrics: dict avec {model_name: metrics_dict}
    :param file_name: nom du fichier de sortie dans results_dir
    """
    file_path = os.path.join(results_dir, file_name)
    with open(file_path, "w") as f:
        for model_name, metrics in models_metrics.items():
            f.write(f"===== {model_name} =====\n\n")
            for key, value in metrics.items():
                if  key not in ["roc_curve"]:
                    f.write(f"{key} :\n{value}\n\n")
                f.write(f"{key} :\n{value}\n\n")
            f.write("="*50 + "\n\n")
    print(f"Toutes les métriques sauvegardées dans {file_path}")

