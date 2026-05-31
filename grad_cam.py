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

    def get_base_model(self):
        """
        Encontra o MobileNetV2
        dentro do modelo principal.
        """
        for layer in self.model.layers:
            if isinstance(
                layer,
                tf.keras.Model
            ):
                return layer

        raise ValueError(
            "Base model não encontrada."
        )

    def get_last_conv_layer_name(
        self,
        base_model
    ):
        """
        Busca automaticamente
        a última Conv2D.
        """
        for layer in reversed(
            base_model.layers
        ):
            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            ):
                return layer.name

        raise ValueError(
            "Nenhuma camada Conv2D encontrada."
        )

    def load_image(
        self,
        image_path
    ):
        image = tf.keras.utils.load_img(
            image_path,
            target_size=self.img_size
        )

        image_array = (
            tf.keras.utils
            .img_to_array(image)
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    def generate_heatmap(
        self,
        image_array
    ):
        base_model = (
            self.get_base_model()
        )

        last_conv_name = (
            self.get_last_conv_layer_name(
                base_model
            )
        )

        print(
            f"Última camada conv: "
            f"{last_conv_name}"
        )

        last_conv_layer = (
            base_model.get_layer(
                last_conv_name
            )
        )

        grad_model = (
            tf.keras.models.Model(
                inputs=self.model.input,
                outputs=[
                    last_conv_layer.output,
                    self.model.output
                ]
            )
        )

        image_tensor = tf.convert_to_tensor(
            image_array,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = (
                grad_model(
                    image_tensor,
                    training=False
                )
            )

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
            conv_outputs
            * pooled_gradients,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        heatmap /= (
            tf.reduce_max(
                heatmap
            ) + 1e-8
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

        saved_count = 0

        for class_name in self.class_names:

            indexes = np.where(
                y_test == class_name
            )[0]

            selected_indexes = (
                indexes[
                    :samples_per_class
                ]
            )

            for idx_number, idx in enumerate(
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
                    f"Real: "
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

                save_path = (
                    os.path.join(
                        self.output_dir,
                        f"{class_name}_"
                        f"{idx_number}.png"
                    )
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
            f"imagens salvas em:"
        )

        print(
            self.output_dir
        )