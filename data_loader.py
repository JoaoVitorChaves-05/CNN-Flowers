import kagglehub
from pathlib import Path
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

class DataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.classes = None
        self.image_dir = None
        self.image_paths = []
        self.labels = []

        self.splited_data = {
            "x_train": None,
            "x_test": None,
            "y_train": None,
            "y_test": None,
            "x_val": None,
            "y_val": None,
        }

    def load_data(self):
        path = kagglehub.dataset_download(
            handle=self.dataset_name
        )

        path = Path(path)

        self.image_dir = join(path, "flower_photos")

        self.classes = sorted(
            folder.name
            for folder in Path(self.image_dir).iterdir()
            if folder.is_dir()
        )

        print(f"Dataset path: {self.image_dir}")
        print(f"Classes encontradas: {self.classes}")

        return self.image_dir

    def get_image_paths_and_labels(self):
        if self.classes is None:
            raise ValueError(
                "Data not loaded. Call load_data() first."
            )

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = join(
                self.image_dir,
                class_name
            )

            for img_path in Path(class_dir).glob("*.jpg"):
                self.image_paths.append(str(img_path))
                self.labels.append(class_name)

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

        print(
            f"Total de imagens: {len(self.image_paths)}"
        )

        return self.image_paths, self.labels

    def split_data(
        self,
        random_state=42
    ):
        """
        70% treino
        15% validação
        15% teste
        """

        # 70 treino / 30 restante
        (
            x_train,
            x_temp,
            y_train,
            y_temp,
        ) = train_test_split(
            self.image_paths,
            self.labels,
            test_size=0.30,
            stratify=self.labels,
            random_state=random_state,
        )

        # 15 validação / 15 teste
        (
            x_val,
            x_test,
            y_val,
            y_test,
        ) = train_test_split(
            x_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=random_state,
        )

        self.splited_data = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "x_val": x_val,
            "y_val": y_val,
        }

        print("\nDivisão dos dados:")
        print(
            f"Treino: {len(x_train)} "
            f"({len(x_train)/len(self.image_paths):.1%})"
        )
        print(
            f"Validação: {len(x_val)} "
            f"({len(x_val)/len(self.image_paths):.1%})"
        )
        print(
            f"Teste: {len(x_test)} "
            f"({len(x_test)/len(self.image_paths):.1%})"
        )

        return self.splited_data
    
    def process_image(
        self,
        image_path,
        label,
        img_size=(224, 224)
    ):
        image = tf.io.read_file(
            image_path
        )

        image = tf.image.decode_jpeg(
            image,
            channels=3
        )

        image = tf.image.resize(
            image,
            img_size
        )

        image = tf.cast(
            image,
            tf.float32
        )

        label = tf.argmax(
            tf.cast(
                label == self.classes,
                tf.int32
            )
        )

        return image, label
    
    def create_dataset(
        self,
        x,
        y,
        batch_size=32,
        training=False,
        img_size=(224, 224),
        data_augmentation=None
    ):
        dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)
        )

        if training:
            dataset = dataset.shuffle(
                buffer_size=1000,
                seed=42
            )

        dataset = dataset.map(
            lambda img, label:
            self.process_image(
                img,
                label,
                img_size
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Data augmentation só no treino
        if training and data_augmentation:
            dataset = dataset.map(
                lambda x, y:
                (
                    data_augmentation(
                        x,
                        training=True
                    ),
                    y
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(
            batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset