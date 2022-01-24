import tensorflow as tf
from pandas import DataFrame
from pathlib import Path
from sklearn import model_selection
from tensorflow.keras import preprocessing
from .config import IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE


def tfdata_generator(datagen: preprocessing.image.ImageDataGenerator) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(lambda: datagen, output_types=(tf.float32, tf.float32))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def preprocess(members: dict[str, list[Path]]) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    filenames: list[str] = []
    categories: list[str] = []
    for key, val in members.items():
        filenames.extend(list(map(lambda v: str(v), val)))
        categories.extend([key] * len(val))
    df = (
        DataFrame(
            {
                "filename": filenames,
                "category": categories,
            }
        )
        .sample(frac=1)
        .reset_index(drop=True)
    )

    train_df, val_df = model_selection.train_test_split(df, test_size=0.1, random_state=420)
    train_df: DataFrame = train_df.reset_index(drop=True)
    val_df: DataFrame = val_df.reset_index(drop=True)

    train_datagen = preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filename",
        y_col="category",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_datagen = preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col="filename",
        y_col="category",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    train_ds = tfdata_generator(train_gen)
    val_ds = tfdata_generator(val_gen)

    return train_ds, val_ds
