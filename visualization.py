import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    roc_curve,
    auc
)

from sklearn.preprocessing import (
    label_binarize
)

class Visualization:
    def __init__(
        self,
        output_dir="output"
    ):
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    def save_training_curves(
        self,
        history
    ):
        """
        Item (d):
        Accuracy/Loss treino e validação
        """

        history_dict = history.history

        train_acc = (
            history_dict["accuracy"]
        )

        val_acc = (
            history_dict["val_accuracy"]
        )

        train_loss = (
            history_dict["loss"]
        )

        val_loss = (
            history_dict["val_loss"]
        )

        epochs = range(
            1,
            len(train_acc) + 1
        )

        # ===== Accuracy =====
        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            train_acc,
            label="Train Accuracy"
        )

        plt.plot(
            epochs,
            val_acc,
            label="Validation Accuracy"
        )

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(
            "Training and Validation Accuracy"
        )
        plt.legend()
        plt.grid(True)

        accuracy_path = os.path.join(
            self.output_dir,
            "accuracy_curve.png"
        )

        plt.savefig(
            accuracy_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # ===== Loss =====
        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            train_loss,
            label="Train Loss"
        )

        plt.plot(
            epochs,
            val_loss,
            label="Validation Loss"
        )

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(
            "Training and Validation Loss"
        )

        plt.legend()
        plt.grid(True)

        loss_path = os.path.join(
            self.output_dir,
            "loss_curve.png"
        )

        plt.savefig(
            loss_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nGráficos salvos em:"
        )

        print(accuracy_path)
        print(loss_path)
        
    def save_confusion_matrix(
        self,
        y_true,
        y_pred,
        class_names
    ):
        """
        Item (e):
        Matriz de confusão
        com números absolutos
        e percentuais.
        """

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        # Percentual por linha
        cm_percent = (
            cm.astype("float")
            / cm.sum(axis=1)[:, np.newaxis]
        ) * 100

        annotations = np.empty_like(
            cm
        ).astype(str)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):

                count = cm[i, j]
                percent = cm_percent[i, j]

                annotations[i, j] = (
                    f"{count}\n"
                    f"({percent:.1f}%)"
                )

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=annotations,
            fmt="",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.xlabel(
            "Predicted Class"
        )

        plt.ylabel(
            "True Class"
        )

        plt.title(
            "Confusion Matrix"
        )

        confusion_path = os.path.join(
            self.output_dir,
            "confusion_matrix.png"
        )

        plt.savefig(
            confusion_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nMatriz de confusão salva:"
        )

        print(confusion_path)
        
    def save_roc_auc(
        self,
        y_true,
        y_prob,
        class_names
    ):
        """
        Item (f):
        Curvas ROC/AUC par a par
        e todas no mesmo gráfico.
        """

        y_true_bin = label_binarize(
            y_true,
            classes=range(len(class_names))
        )

        pair_indexes = [
            (0, 1),  # daisy x dandelion
            (1, 2),  # dandelion x roses
            (2, 3),  # roses x sunflowers
            (3, 4),  # sunflowers x tulips
            (4, 0),  # tulips x daisy
        ]

        plt.figure(figsize=(10, 8))

        for i, j in pair_indexes:
            # Seleciona apenas amostras das duas classes
            mask = np.logical_or(
                y_true == i,
                y_true == j
            )

            y_binary = (
                y_true[mask] == j
            ).astype(int)

            y_scores = y_prob[
                mask,
                j
            ]

            fpr, tpr, _ = roc_curve(
                y_binary,
                y_scores
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            plt.plot(
                fpr,
                tpr,
                linewidth=2,
                label=(
                    f"{class_names[i]} "
                    f"vs "
                    f"{class_names[j]} "
                    f"(AUC={roc_auc:.3f})"
                )
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC/AUC Curves"
        )

        plt.legend()

        plt.grid(True)

        roc_path = os.path.join(
            self.output_dir,
            "roc_auc.png"
        )

        plt.savefig(
            roc_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nROC/AUC salvo:"
        )

        print(roc_path)
        
    def save_class_accuracy_errors(
        self,
        y_true,
        y_pred,
        class_names
    ):
        """
        Item (g):
        Gráfico de acertos e erros
        por classe.
        """

        correct_counts = []
        incorrect_counts = []

        correct_percentages = []
        incorrect_percentages = []

        for idx in range(
            len(class_names)
        ):
            mask = (
                y_true == idx
            )

            total = np.sum(mask)

            correct = np.sum(
                y_pred[mask] == idx
            )

            incorrect = (
                total - correct
            )

            correct_counts.append(
                correct
            )

            incorrect_counts.append(
                incorrect
            )

            correct_percentages.append(
                (correct / total) * 100
            )

            incorrect_percentages.append(
                (incorrect / total) * 100
            )

        x = np.arange(
            len(class_names)
        )

        width = 0.35

        plt.figure(figsize=(12, 7))

        bars_correct = plt.bar(
            x - width / 2,
            correct_counts,
            width,
            label="Correct"
        )

        bars_incorrect = plt.bar(
            x + width / 2,
            incorrect_counts,
            width,
            label="Incorrect"
        )

        for i, bar in enumerate(
            bars_correct
        ):
            plt.text(
                bar.get_x()
                + bar.get_width()/2,
                bar.get_height(),
                (
                    f"{correct_counts[i]}"
                    f"\n"
                    f"({correct_percentages[i]:.1f}%)"
                ),
                ha="center"
            )

        for i, bar in enumerate(
            bars_incorrect
        ):
            plt.text(
                bar.get_x()
                + bar.get_width()/2,
                bar.get_height(),
                (
                    f"{incorrect_counts[i]}"
                    f"\n"
                    f"({incorrect_percentages[i]:.1f}%)"
                ),
                ha="center"
            )

        plt.xticks(
            x,
            class_names
        )

        plt.ylabel(
            "Quantidade"
        )

        plt.title(
            "Correct vs Incorrect Predictions per Class"
        )

        plt.legend()

        plt.grid(
            axis="y"
        )

        class_path = os.path.join(
            self.output_dir,
            "class_accuracy_errors.png"
        )

        plt.savefig(
            class_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nGráfico de acertos/erros salvo:"
        )

        print(class_path)