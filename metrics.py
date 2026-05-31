import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class MetricsEvaluator:
    def __init__(
        self,
        model,
        class_names
    ):
        self.model = model
        self.class_names = class_names

        self.y_true = []
        self.y_pred = []
        self.y_prob = None

    def predict(
        self,
        test_ds
    ):
        """
        Realiza predição no conjunto de teste
        e armazena resultados para métricas.
        """

        self.y_true = []

        for _, labels in test_ds:
            self.y_true.extend(
                labels.numpy()
            )

        self.y_true = np.array(
            self.y_true
        )

        # Probabilidades
        self.y_prob = (
            self.model.predict(
                test_ds,
                verbose=1
            )
        )

        # Classe prevista
        self.y_pred = np.argmax(
            self.y_prob,
            axis=1
        )

        return (
            self.y_true,
            self.y_pred,
            self.y_prob
        )

    def classification_metrics(self):
        """
        Calcula métricas de classificação.
        """

        accuracy = accuracy_score(
            self.y_true,
            self.y_pred
        )

        precision = precision_score(
            self.y_true,
            self.y_pred,
            average="weighted"
        )

        recall = recall_score(
            self.y_true,
            self.y_pred,
            average="weighted"
        )

        f1 = f1_score(
            self.y_true,
            self.y_pred,
            average="weighted"
        )

        print("\n===== MÉTRICAS =====")

        print(
            f"Acurácia: "
            f"{accuracy:.4f}"
        )

        print(
            f"Precisão: "
            f"{precision:.4f}"
        )

        print(
            f"Recall: "
            f"{recall:.4f}"
        )

        print(
            f"F1-Score: "
            f"{f1:.4f}"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }