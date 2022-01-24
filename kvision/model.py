from loguru import logger
from tensorflow.keras import layers, models, Model
from .config import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS


def get_model() -> Model:
    conv_layers = []
    for i, f in enumerate([32, 64, 128, 128]):
        conv_layers.extend(
            [
                layers.Conv2D(
                    filters=f,
                    kernel_size=3,
                    activation="relu",
                    name=f"conv{i + 1}",
                ),
                layers.MaxPooling2D(
                    pool_size=3,
                    strides=2,
                    name=f"pool{i + 1}",
                ),
            ]
        )

    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
            *conv_layers,
            layers.Flatten(name="flatten"),
            layers.Dropout(0.5, name="drophalf"),
            layers.Dense(512, activation="relu", name="fc1"),
            layers.Dense(5, activation="sigmoid", name="output"),
        ],
        name="kvision",
    )
    logger.info(f"\n{model.summary()}")
    return model
