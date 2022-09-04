import csv
import logging
import os

import tensorflow as tf
from keras import Model, mixed_precision
from keras.callbacks import (
    Callback,
    CSVLogger,
    History,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
)
from keras.initializers.initializers_v2 import RandomNormal, Zeros
from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
)
from keras.losses import Loss, SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import SGD, Optimizer
from keras.utils import image_dataset_from_directory
from loguru import logger
from matplotlib import pyplot as plt
from tensorflow_addons.callbacks import TQDMProgressBar

from .config import (
    BASE_DIR,
    BATCH_SIZE,
    DATASET_PATH,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


@logger.catch()
class KVision:
    def __init__(self):
        os.makedirs(BASE_DIR / "kvision" / "run", exist_ok=True)
        logger_ = tf.get_logger()
        logger_.setLevel(logging.DEBUG)
        mixed_precision.set_global_policy(mixed_precision.Policy("mixed_float16"))
        print(tf.config.experimental.list_physical_devices())
        plt.style.use("seaborn")

        self.image_height: int = IMAGE_HEIGHT
        self.image_width: int = IMAGE_WIDTH
        self.image_channels: int = IMAGE_CHANNELS
        self.batch_size: int = BATCH_SIZE
        self.seed = 314
        self.epochs = 100
        self.class_names: list[str] = []
        self.num_classes = 0

    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        ds = [
            image_dataset_from_directory(
                DATASET_PATH,
                validation_split=0.2,
                subset=subset,
                seed=self.seed,
                image_size=(self.image_height, self.image_width),
                batch_size=self.batch_size,
            )
            for subset in ["training", "validation"]
        ]
        train_ds, val_ds = ds
        self.class_names = train_ds.class_names
        self.num_classes = len(self.class_names)
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_ds, val_ds

    @staticmethod
    def Conv2D_(filters: int, kernel_size: int, strides: int = 1, name: str = None, *args, **kwargs):
        return Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            name=name,
            activation="relu",
            use_bias=True,
            bias_initializer=Zeros(),
            kernel_initializer=RandomNormal(mean=0, stddev=1e-2),
            *args,
            **kwargs,
        )

    @staticmethod
    def Dense_(units: int, name: str = None, relu=False, *args, **kwargs):
        return Dense(
            units=units,
            name=name,
            activation="relu" if relu else None,
            use_bias=True,
            bias_initializer=Zeros(),
            kernel_initializer=RandomNormal(mean=0, stddev=1e-2),
            *args,
            **kwargs,
        )

    def get_model(self) -> Model:
        Augmentation = lambda: Sequential(
            [
                RandomFlip("horizontal"),
                RandomRotation(0.1),
                RandomZoom(0.1),
            ],
            name="augmentation",
        )

        model = Sequential(
            [
                Augmentation(),
                Rescaling(scale=1 / 255, name="rescale"),
                self.Conv2D_(filters=96, kernel_size=7, strides=2, name="conv1"),
                MaxPooling2D(pool_size=3, strides=2, name="pool1"),
                self.Conv2D_(filters=256, kernel_size=5, strides=2, name="conv2"),
                MaxPooling2D(pool_size=3, strides=2, name="pool2"),
                self.Conv2D_(filters=384, kernel_size=3, name="conv3"),
                self.Conv2D_(filters=384, kernel_size=3, name="conv4"),
                self.Conv2D_(filters=256, kernel_size=3, name="conv5"),
                MaxPooling2D(pool_size=3, strides=2, name="pool3"),
                Flatten(name="flatten"),
                self.Dense_(4096, relu=True, name="fc1"),
                Dropout(0.5),
                self.Dense_(4096, relu=True, name="fc2"),
                Dropout(0.5),
                self.Dense_(self.num_classes, name="output"),
            ],
            name="kvision-0.1.0",
        )
        return model

    @staticmethod
    def get_callbacks() -> list[Callback]:
        tqdm = TQDMProgressBar(leave_epoch_progress=False)
        history = History()
        history.set_params({"verbose": 0})
        checkpoint = ModelCheckpoint(
            BASE_DIR / "kvision" / "run" / "kvision-0.1.0.h5",
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            mode="min",
        )
        csv_logger = CSVLogger(
            BASE_DIR / "kvision" / "run" / "training.csv",
            separator=",",
            append=False,
        )
        terminate_nan = TerminateOnNaN()
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1)
        return [
            history,
            tqdm,
            checkpoint,
            csv_logger,
            terminate_nan,
            reduce_lr,
        ]

    @staticmethod
    def get_optimizer() -> Optimizer | str:
        return SGD(learning_rate=1e-2, momentum=0.9)

    @staticmethod
    def get_loss() -> Loss | str:
        return SparseCategoricalCrossentropy(from_logits=True)

    def initialize_model(self) -> Model:
        model = self.get_model()
        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.get_loss(),
            metrics=["accuracy"],
        )
        model.build(input_shape=(None, self.image_height, self.image_width, self.image_channels))
        model.summary()
        return model

    @staticmethod
    def save_metric_graphs(epochs: list[int], metrics: dict[str, list[float]]):
        plt.plot(epochs, metrics["loss"], label="train")
        plt.plot(epochs, metrics["val_loss"], label="val")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(BASE_DIR / "kvision" / "run" / "loss.png", bbox_inches="tight", dpi=144)
        plt.close("all")

        plt.plot(epochs, metrics["accuracy"], label="train")
        plt.plot(epochs, metrics["val_accuracy"], label="val")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(BASE_DIR / "kvision" / "run" / "accuracy.png", bbox_inches="tight", dpi=144)
        plt.close("all")

    @staticmethod
    def evaluate_model(model: Model, val_ds: tf.data.Dataset):
        eva = model.evaluate(val_ds.take(1))
        logger.info(f"Loss: {eva[0]}, Accuracy: {eva[1] * 100}%")

    def __call__(self):
        train_ds, val_ds = self.load_data()
        model = self.initialize_model()
        callbacks = self.get_callbacks()
        try:
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=0,
            )
        except KeyboardInterrupt:
            model.save(BASE_DIR / "kvision" / "run" / "kvision-0.1.0-model")
            with open(BASE_DIR / "kvision" / "run" / "training.csv", "r") as f:
                reader = csv.DictReader(f)
                epochs = [r["epoch"] for r in reader]
                metrics = {
                    "loss": [],
                    "val_loss": [],
                    "accuracy": [],
                    "val_accuracy": [],
                }
                for row in reader:
                    for k in metrics.keys():
                        metrics[k].append(row[k])
            self.save_metric_graphs(epochs, metrics)
        else:
            self.save_metric_graphs(history.epoch, history.history)
            self.evaluate_model(model, val_ds)


if __name__ == "__main__":
    KVision()()
