from loguru import logger
from .load import load
from .preprocess import preprocess
from .model import get_model


@logger.catch()
def main() -> None:
    train_ds, val_ds = preprocess(load())
    model = get_model()


if __name__ == "__main__":
    main()
