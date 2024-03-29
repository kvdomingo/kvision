import logging
import os
import sys
from datetime import datetime

import tensorflow as tf
from keras import Model, mixed_precision
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    History,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    TerminateOnNaN,
)
from keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Input,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from keras.losses import Loss, SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adagrad, Optimizer
from keras.utils import image_dataset_from_directory
from loguru import logger
from matplotlib import pyplot as plt
from numpy import array
from tensorflow_addons.callbacks import TQDMProgressBar

from .config import BASE_DIR, BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


@logger.catch()
class KVision:
    model: Model
    history: History
    validation_dataset: tf.data.Dataset
    class_names: list[str]
    num_classes: int

    def __init__(self):
        os.makedirs(BASE_DIR / "kvision" / "run", exist_ok=True)
        logger_ = tf.get_logger()
        logger_.setLevel(logging.DEBUG)
        mixed_precision.set_global_policy(mixed_precision.Policy("mixed_float16"))
        print(*tf.config.experimental.list_physical_devices(), sep="\n")
        plt.style.use("seaborn")

        self.image_height: int = IMAGE_HEIGHT
        self.image_width: int = IMAGE_WIDTH
        self.image_channels: int = IMAGE_CHANNELS
        self.batch_size: int = BATCH_SIZE
        self.seed = 314
        self.epochs = 100

    def load_data(self, training: bool = True) -> tuple[tf.data.Dataset | None, tf.data.Dataset | None]:
        ds = [
            image_dataset_from_directory(
                BASE_DIR / "data",
                validation_split=0.2,
                subset=subset,
                seed=self.seed,
                image_size=(self.image_height, self.image_width),
                batch_size=self.batch_size,
            )
            for subset in ["training", "validation"]
        ]
        train_ds, val_ds = ds
        self.validation_dataset = val_ds
        self.class_names = train_ds.class_names
        self.num_classes = len(self.class_names)
        if training:
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_ds, val_ds

    @staticmethod
    def Augmentation():
        return Sequential(
            [
                RandomRotation(factor=0.1),
                RandomZoom(height_factor=0.1, width_factor=0.1),
                RandomFlip(mode="horizontal"),
            ],
            name="augmentation",
        )

    def get_model(self, training: bool = True) -> Model:
        inception = InceptionResNetV2(include_top=False, weights="imagenet")
        for i in range(len(inception.layers)):
            inception.layers[i].trainable = i > 100

        layers = [
            Input(shape=(self.image_height, self.image_width, self.image_channels)),
            inception,
            GlobalAveragePooling2D(name="avg_pool"),
            Dense(units=512, activation="relu", name="fc1"),
            Dense(self.num_classes, name="output"),
        ]
        if training:
            layers.insert(1, self.Augmentation())
        model = Sequential(layers, name="kvision-0.1.0")
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
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10)
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-5,
            patience=15,
            verbose=0,
            restore_best_weights=True,
        )
        tensorboard = TensorBoard(
            log_dir=BASE_DIR / "kvision" / "run" / "tb_logs" / datetime.now().strftime("%Y-%m-%d %H%MH")
        )
        return [
            history,
            tqdm,
            checkpoint,
            csv_logger,
            terminate_nan,
            reduce_lr,
            early_stop,
            tensorboard,
        ]

    @staticmethod
    def get_optimizer() -> Optimizer | str:
        return Adagrad(learning_rate=0.1)

    @staticmethod
    def get_loss() -> Loss | str:
        return SparseCategoricalCrossentropy(from_logits=True)

    def initialize_model(self, training: bool = True):
        model = self.get_model(training)
        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.get_loss(),
            metrics=["accuracy"],
        )
        model.build(input_shape=(None, self.image_height, self.image_width, self.image_channels))
        self.model = model
        if not training:
            return
        model.summary()
        if str(input("Proceed? ([y]/n) ")).lower().strip() == "n":
            sys.exit(1)

    def save_metric_graphs(self, epochs: list[int] = None, metrics: dict[str, list[float]] = None):
        if epochs is None:
            epochs = self.history.epoch
        if metrics is None:
            metrics = self.history.history
        epochs = array(epochs, dtype="uint8")
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

    def evaluate_model(self, val_ds: tf.data.Dataset = None):
        if val_ds is None:
            val_ds = self.validation_dataset
        eva = self.model.evaluate(val_ds.take(1))
        logger.info(f"Loss: {eva[0]}, Accuracy: {eva[1] * 100}%")

    def __call__(self):
        train_ds, val_ds = self.load_data()
        self.initialize_model()
        callbacks = self.get_callbacks()
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=0,
        )
        self.save_metric_graphs()
        self.evaluate_model()
