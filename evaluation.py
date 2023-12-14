import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def get_cm_sklearn(y_eval, y_pred_eval, *, labels=None):
    _, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs = axs.flatten()

    titles = [
        "\nnormalizada pelos valores reais",
        "\nnormalizada pelos valores preditos",
        "\nnormalizada pelos pela população",
        "",
    ]
    normalizes = ["true", "pred"]

    for ax, title, normalize in zip(axs, titles, normalizes):
        cm = confusion_matrix(y_eval, y_pred_eval, normalize=normalize)
        sns.set(font_scale=0.9)
        sns.heatmap(
            cm,
            cmap="YlGnBu",
            linewidths=0.5,
            fmt=".1%" if normalize else ".0f",
            annot=True,
            cbar=False,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.set(
            xlabel="Predição", ylabel="Valor Real", title=f"Matriz de confusão{title}"
        )
    plt.show()


def print_metrics(y_eval, y_pred_eval):
    print(f"Accuracy: {accuracy_score(y_eval, y_pred_eval)}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_eval, y_pred_eval)}")
    print(f"F1: {f1_score(y_eval, y_pred_eval)}")
    print(f"RoC AuC: {roc_auc_score(y_eval, y_pred_eval)}")
    RocCurveDisplay.from_predictions(y_eval, y_pred_eval)
