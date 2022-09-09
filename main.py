from click import argument, group


@group()
def cli():
    pass


@cli.command()
def train():
    import csv

    from kvision import BASE_DIR, KVision

    kvision = KVision()
    try:
        kvision()
    except KeyboardInterrupt:
        kvision.model.save(BASE_DIR / "kvision" / "run" / "kvision-0.1.0-model")
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
        kvision.save_metric_graphs(epochs, metrics)
    else:
        kvision.save_metric_graphs()
        kvision.evaluate_model()


@cli.command()
def evaluate():
    from kvision.evaluate import evaluate

    evaluate()


@cli.command()
@argument("filename")
def predict(filename: str):
    from pathlib import Path

    from kvision.predict import predict

    predict(Path(filename).resolve())


if __name__ == "__main__":
    cli()
