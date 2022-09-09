from os import PathLike

import tensorflow as tf
from numpy import argmax, max

from . import BASE_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, KVision


def predict(filename: PathLike):
    raw = tf.io.read_file(str(filename))
    image = tf.image.decode_image(raw, channels=3, dtype="float16")
    image = tf.multiply(image, 1 / 255)
    image = tf.image.resize([image], [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.LANCZOS5)[0]
    image = tf.expand_dims(image, 0)

    kvision = KVision()
    kvision.load_data(training=False)
    kvision.initialize_model(training=False)
    kvision.model.load_weights(BASE_DIR / "kvision" / "run" / "kvision-0.1.0.h5")
    prediction = kvision.model(image)
    score = tf.nn.relu(prediction[0])
    print(kvision.class_names[argmax(score)], max(score))
