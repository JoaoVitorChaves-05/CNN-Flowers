from data_loader import DataLoader
from model import CNNModel
from trainer import Trainer
from metrics import MetricsEvaluator
from visualization import Visualization
from grad_cam import GradCAM
from model_manager import ModelManager

import tensorflow as tf
import numpy as np

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    dataset_name = (
        "rahmasleam/flowers-dataset"
    )

    # ===== Item (a) =====
    # Carregamento do dataset
    data_loader = DataLoader(
        dataset_name
    )

    data_loader.load_data()

    data_loader.get_image_paths_and_labels()

    splited_data = (
        data_loader.split_data()
    )

    # ===== Data augmentation =====
    data_augmentation = (
        tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(
                    "horizontal"
                ),
                tf.keras.layers.RandomRotation(
                    0.1
                ),
                tf.keras.layers.RandomZoom(
                    0.1
                ),
            ]
        )
    )

    # ===== Datasets =====
    train_ds = (
        data_loader.create_dataset(
            splited_data["x_train"],
            splited_data["y_train"],
            training=True,
            data_augmentation=data_augmentation
        )
    )

    val_ds = (
        data_loader.create_dataset(
            splited_data["x_val"],
            splited_data["y_val"]
        )
    )

    test_ds = (
        data_loader.create_dataset(
            splited_data["x_test"],
            splited_data["y_test"]
        )
    )

    print(
        "\nDatasets criados!"
    )

    # ===== Item (b) =====
    model_manager = (
        ModelManager()
    )

    model = (
        model_manager
        .load_model()
    )

    history = None
    training_time = 0

    if model is None:

        print(
            "\nNenhum modelo salvo encontrado."
        )

        print(
            "Treinando novo modelo..."
        )

        cnn_model = CNNModel(
            num_classes=len(
                data_loader.classes
            )
        )

        model = (
            cnn_model.build_model()
        )

        cnn_model.summary()

        # ===== Item (h) =====
        trainer = Trainer(
            model=model,
            epochs=20
        )

        (
            history,
            training_time
        ) = trainer.train(
            train_ds,
            val_ds
        )

        model_manager.save_model(
            model
        )

    else:

        print(
            "\nModelo carregado!"
        )

        model.summary()

    # ===== Item (c) =====
    metrics_evaluator = (
        MetricsEvaluator(
            model=model,
            class_names=data_loader.classes
        )
    )

    (
        y_true,
        y_pred,
        y_prob
    ) = metrics_evaluator.predict(
        test_ds
    )

    metrics = (
        metrics_evaluator
        .classification_metrics()
    )

    # ===== Item (d) =====
    visualization = (
        Visualization()
    )

    if history is not None:
        visualization.save_training_curves(
            history
        )

    # ===== Item (e) =====
    visualization.save_confusion_matrix(
        y_true,
        y_pred,
        data_loader.classes
    )

    # ===== Item (f) =====
    visualization.save_roc_auc(
        y_true,
        y_prob,
        data_loader.classes
    )

    # ===== Item (g) =====
    visualization.save_class_accuracy_errors(
        y_true,
        y_pred,
        data_loader.classes
    )

    # ===== Item (i) =====
    grad_cam = (
        GradCAM(
            model=model,
            class_names=data_loader.classes
        )
    )

    grad_cam.generate(
        x_test=splited_data["x_test"],
        y_test=splited_data["y_test"],
        samples_per_class=3
    )

    print(
        "\nExecução finalizada!"
    )

    print(
        f"Tempo total de treino: "
        f"{training_time:.2f} minutos"
    )

    print(
        "\nItens concluídos:"
    )

    print("✔ Item (a)")
    print("✔ Item (b)")
    print("✔ Item (c)")
    print("✔ Item (d)")
    print("✔ Item (e)")
    print("✔ Item (f)")
    print("✔ Item (g)")
    print("✔ Item (h)")
    print("✔ Item (i)")


if __name__ == "__main__":
    main()