from shap import GradientExplainer
from torch import concat, save
import polars as pl
from abcd.model import make_trainer
from abcd.preprocess import EVENT_MAPPING


RACE_MAPPING = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
SEX_MAPPING = {1: "Male", 2: "Female"}


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


def r2_results(y_pred, y_true):
    df = pl.read_csv("data/analytic/test.csv", columns=["src_subject_id"])
    df = df.with_columns(
        pl.cum_count().over("src_subject_id").cast(pl.Utf8).alias("eventname"),
        y_pred=y_pred,
        y_true=y_true,
    )
    demographics = pl.read_csv("data/demographics.csv").with_columns(
        pl.col("eventname").replace(EVENT_MAPPING),
        pl.col("race_ethnicity").replace(RACE_MAPPING),
        pl.col("sex").replace(SEX_MAPPING),
    )
    df = df.join(demographics, on=["src_subject_id", "eventname"], how="inner").sort(
        ["src_subject_id", "eventname"]
    )
    df.write_csv("data/results/predicted_vs_observed.csv")


def make_shap_values(model, data_module):
    test_dataloader = iter(data_module.test_dataloader())
    X, _ = next(test_dataloader)
    background, _ = next(test_dataloader)
    explainer = GradientExplainer(model, background.to("mps:0"))
    shap_values = explainer.shap_values(X.to("mps:0"))
    save(shap_values, "data/results/shap_values.pt")


def evaluate(config, model, data_module):
    # make_shap_values(model, data_module)
    y_pred, y_true = get_predictions(config, model, data_module)
    r2_results(y_pred=y_pred, y_true=y_true)
