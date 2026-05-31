import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(
        self,
        model,
        class_names,
        output_dir="output/gradcam",
        img_size=(224, 224)
    ):
        self.model = model
        self.class_names = class_names
        self.output_dir = output_dir
        self.img_size = img_size

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.base_model = (
            self._find_base_model()
        )

        self.last_conv_layer = (
            self._find_last_conv_layer()
        )

        self.inner_grad_model = (
            tf.keras.models.Model(
                inputs=self.base_model.inputs,
                outputs=[
                    self.last_conv_layer.output,
                    self.base_model.output
                ]
            )
        )

        print(
            f"\nBase model: "
            f"{self.base_model.name}"
        )

        print(
            f"Ultima camada conv: "
            f"{self.last_conv_layer.name}"
        )

    def _find_base_model(self):
        """
        Busca o modelo base dentro do modelo principal.
        """
        for layer in self.model.layers:
            if isinstance(
                layer,
                tf.keras.Model
            ):
                return layer

        raise ValueError(
            "Base model nao encontrado."
        )

    def _find_last_conv_layer(self):
        """
        Busca a ultima camada Conv2D dentro do base model.
        """
        for layer in reversed(
            self.base_model.layers
        ):
            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            ):
                return layer

        raise ValueError(
            "Nenhuma Conv2D encontrada no base model."
        )

    def load_image(
        self,
        image_path
    ):
        image = tf.keras.utils.load_img(
            image_path,
            target_size=self.img_size
        )

        image = (
            tf.keras.utils
            .img_to_array(image)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        return image

    def generate_heatmap(
        self,
        image_array
    ):
        image_tensor = tf.convert_to_tensor(
            image_array,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:
            x = image_tensor

            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue

                if layer.name == self.base_model.name:
                    conv_outputs, x = self.inner_grad_model(
                        x,
                        training=False
                    )
                else:
                    try:
                        x = layer(
                            x,
                            training=False
                        )
                    except TypeError:
                        x = layer(x)

            predictions = x
            predicted_class = tf.argmax(
                predictions[0]
            )

            loss = predictions[
                :,
                predicted_class
            ]

        gradients = tape.gradient(
            loss,
            conv_outputs
        )

        if gradients is None:
            raise ValueError(
                "Nao foi possivel calcular os gradientes."
            )

        pooled_gradients = (
            tf.reduce_mean(
                gradients,
                axis=(0, 1, 2)
            )
        )

        conv_outputs = (
            conv_outputs[0]
        )

        heatmap = tf.reduce_sum(
            pooled_gradients
            * conv_outputs,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        heatmap = (
            heatmap
            / (
                tf.reduce_max(
                    heatmap
                ) + 1e-8
            )
        )

        return (
            heatmap.numpy(),
            predicted_class.numpy()
        )

    def overlay_heatmap(
        self,
        image_path,
        heatmap
    ):
        image = cv2.imread(
            image_path
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        heatmap = cv2.resize(
            heatmap,
            (
                image.shape[1],
                image.shape[0]
            )
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        superimposed = cv2.addWeighted(
            image,
            0.6,
            heatmap,
            0.4,
            0
        )

        return (
            image,
            superimposed
        )

    def generate(
        self,
        x_test,
        y_test,
        samples_per_class=3
    ):
        print(
            "\nGerando Grad-CAM..."
        )

        x_test = np.array(
            x_test
        )

        y_test = np.array(
            y_test
        )

        # Converte labels texto em indices, se necessario
        if y_test.dtype.kind in {"U", "S", "O"}:
            class_to_index = {
                name: idx
                for idx, name in enumerate(
                    self.class_names
                )
            }

            y_test = np.array(
                [
                    class_to_index[label]
                    for label in y_test
                ]
            )

        saved_count = 0

        for class_index, class_name in enumerate(
            self.class_names
        ):
            indexes = np.where(
                y_test == class_index
            )[0]

            print(
                f"{class_name}: "
                f"{len(indexes)} imagens"
            )

            if len(indexes) == 0:
                continue

            selected_indexes = (
                indexes[
                    :samples_per_class
                ]
            )

            for i, idx in enumerate(
                selected_indexes,
                start=1
            ):
                image_path = (
                    x_test[idx]
                )

                image_array = (
                    self.load_image(
                        image_path
                    )
                )

                (
                    heatmap,
                    predicted_class
                ) = (
                    self.generate_heatmap(
                        image_array
                    )
                )

                (
                    original,
                    gradcam
                ) = (
                    self.overlay_heatmap(
                        image_path,
                        heatmap
                    )
                )

                predicted_label = (
                    self.class_names[
                        predicted_class
                    ]
                )

                plt.figure(
                    figsize=(8, 4)
                )

                plt.subplot(1, 2, 1)

                plt.imshow(
                    original
                )

                plt.title(
                    f"True: "
                    f"{class_name}"
                )

                plt.axis("off")

                plt.subplot(1, 2, 2)

                plt.imshow(
                    gradcam
                )

                plt.title(
                    f"Pred: "
                    f"{predicted_label}"
                )

                plt.axis("off")

                save_path = os.path.join(
                    self.output_dir,
                    f"{class_name}_{i}.png"
                )

                plt.savefig(
                    save_path,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()

                saved_count += 1

        print(
            f"\n{saved_count} "
            f"Grad-CAMs salvos em:"
        )

        print(
            self.output_dir
        )