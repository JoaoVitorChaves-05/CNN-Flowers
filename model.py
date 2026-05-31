import tensorflow as tf


class CNNModel:
    def __init__(
        self,
        num_classes,
        input_shape=(224, 224, 3),
        learning_rate=1e-5
    ):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate

        self.model = None

    def build_model(self):
        base_model = (
            tf.keras.applications.MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet"
            )
        )

        # Fine tuning leve
        base_model.trainable = True

        for layer in base_model.layers[:-30]:
            layer.trainable = False

        inputs = tf.keras.Input(
            shape=self.input_shape
        )

        x = (
            tf.keras.applications
            .mobilenet_v2
            .preprocess_input(inputs)
        )

        x = base_model(
            x,
            training=False
        )

        x = (
            tf.keras.layers
            .GlobalAveragePooling2D()
        )(x)

        x = tf.keras.layers.Dense(
            256,
            activation="relu"
        )(x)

        x = tf.keras.layers.Dropout(
            0.4
        )(x)

        outputs = tf.keras.layers.Dense(
            self.num_classes,
            activation="softmax"
        )(x)

        # <<< FALTAVA ISSO
        self.model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return self.model

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            print(
                "Modelo ainda não foi criado."
            )