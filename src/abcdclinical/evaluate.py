from torch import concat
from sklearn.metrics import r2_score
import polars as pl

from abcdclinical.model import make_trainer
from abcdclinical.preprocess import EVENT_MAPPING


def get_predictions(config, model, data_module):
    trainer, _ = make_trainer(config)
    predictions = trainer.predict(
        model=model, dataloaders=data_module.test_dataloader()
    )
    y_pred, y_true = zip(*predictions)
    y_pred = concat(y_pred)
    y_true = concat(y_true)
    mask = ~y_true.isnan()
    y_pred = y_pred[mask].numpy()
    y_true = y_true[mask].numpy()
    return y_pred.squeeze(1), y_true


def apply_r2(args) -> pl.Series:
    return pl.Series([r2_score(y_true=args[0], y_pred=args[1])], dtype=pl.Float32)


def r2_results(y_pred, y_true):
    df = pl.read_csv("data/analytic/test.csv", columns=["src_subject_id"])
    df = df.with_columns(
        pl.cum_count().over("src_subject_id").cast(pl.Utf8).alias("eventname"),
        y_pred=y_pred,
        y_true=y_true,
    )
    demographics = pl.read_csv("data/demographics.csv").with_columns(
        pl.col("eventname").replace(EVENT_MAPPING)
    )
    df = df.join(demographics, on=["src_subject_id", "eventname"], how="inner").sort(
        ["src_subject_id", "eventname"]
    )
    df.write_csv("data/results/predicted_vs_observed.csv")
    df = df.group_by("eventname").agg(
        pl.apply(exprs=["y_true", "y_pred"], function=apply_r2).alias("r2")
    )
    df.write_csv("data/results/r2_by_event.csv")


def evaluate(config, model, data_module):
    y_pred, y_true = get_predictions(config, model, data_module)
    r2_results(y_pred=y_pred, y_true=y_true)
