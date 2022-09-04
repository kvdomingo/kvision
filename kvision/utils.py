import tensorflow as tf
from keras import preprocessing


def tfdata_generator(datagen: preprocessing.image.ImageDataGenerator) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(lambda: datagen, output_types=(tf.float32, tf.float32))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
