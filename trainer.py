import time


class Trainer:
    def __init__(
        self,
        model,
        epochs=15
    ):
        self.model = model
        self.epochs = epochs

        self.history = None
        self.training_time = None

    def train(
        self,
        train_ds,
        val_ds
    ):
        start_time = time.time()

        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs
        )

        end_time = time.time()

        self.training_time = (
            end_time - start_time
        ) / 60

        print(
            f"\nTempo de treino: "
            f"{self.training_time:.2f} minutos"
        )

        return (
            self.history,
            self.training_time
        )