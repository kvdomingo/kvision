from os import PathLike

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import img_to_array, load_img
from matplotlib.pyplot import Axes, Figure
from numpy import argmax

from . import BASE_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, KVision


def predict(filename: PathLike):
    image = load_img(str(filename), target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image = img_to_array(image, dtype="float16")
    image = tf.expand_dims(image, 0)

    kvision = KVision()
    kvision.load_data(training=False)
    kvision.initialize_model(training=False)
    kvision.model.load_weights(BASE_DIR / "kvision" / "run" / "kvision-0.1.0.h5")
    prediction = kvision.model(image)
    print(kvision.class_names[argmax(prediction[0])])


def predict_batch():
    kvision = KVision()
    kvision.load_data(training=False)
    kvision.initialize_model(training=False)
    kvision.model.load_weights(BASE_DIR / "kvision" / "run" / "kvision-0.1.0.h5")
    for images, _ in kvision.validation_dataset.take(1):
        break
    predictions = kvision.model(images)
    fig: Figure = plt.figure(figsize=(7, 7))
    for i in range(9):
        ax: Axes = fig.add_subplot(3, 3, i + 1)
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(kvision.class_names[argmax(predictions[i])])
        ax.axis("off")
    plt.savefig(BASE_DIR / "kvision" / "run" / "predict-batch.png", dpi=144, bbox_inches="tight")
    plt.close("all")
