import os
import tensorflow as tf


class ModelManager:
    def __init__(
        self,
        model_path="saved_model/flowers_model.keras"
    ):
        self.model_path = model_path

        os.makedirs(
            os.path.dirname(
                self.model_path
            ),
            exist_ok=True
        )

    def save_model(
        self,
        model
    ):
        model.save(
            self.model_path
        )

        print(
            "\nModelo salvo em:"
        )

        print(
            self.model_path
        )

    def load_model(self):
        if os.path.exists(
            self.model_path
        ):
            print(
                "\nCarregando modelo salvo..."
            )

            return (
                tf.keras.models.load_model(
                    self.model_path
                )
            )

        return None