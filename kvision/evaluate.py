from . import BASE_DIR, KVision


def evaluate():
    kvision = KVision()
    kvision.load_data(training=False)
    kvision.initialize_model(training=False)
    kvision.model.load_weights(BASE_DIR / "kvision" / "run" / "kvision-0.1.0.h5")
    kvision.evaluate_model()
