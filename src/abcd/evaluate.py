from shap import GradientExplainer
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch import concat, save, load as torch_load
import polars as pl
from tqdm import tqdm
from abcd.config import Config
from abcd.model import make_trainer
import pandas as pd
import numpy as np

from abcd.tables import SEX_MAPPING

REVERSE_EVENT_MAPPING = {
    0: "Baseline",
    1: "Year 1",
    2: "Year 2",
    3: "Year 3",
    4: "Year 4",
}


def get_predictions(config, model, data_module):
    trainer, _ = make_trainer(config)
    predictions = trainer.predict(
        model=model, dataloaders=data_module.test_dataloader()
    )
    y_pred, y_true = zip(*predictions)
    # FIXME need to not mean when only one target?
    y_pred = concat(y_pred).nanmean(dim=1)
    y_true = concat(y_true).nanmean(dim=1)
    # mask = ~y_true.isnan()
    # y_pred = y_pred[mask].numpy()
    # y_true = y_true[mask].numpy()
    return y_pred, y_true


def r2(y_pred, y_true, config: Config):
    r2s = []
    for _ in range(1000):
        y_pred_resampled, y_true_resampled = resample(y_pred, y_true)
        r2 = r2_score(
            y_true=y_true_resampled, y_pred=y_pred_resampled, multioutput="raw_values"
        )
        r2s.append(r2)
    r2 = pl.DataFrame(r2s, schema=config.labels.cbcl_labels).melt()
    r2.write_csv(f"data/results/r2_{config.target}.csv")


def r2_results(y_pred, y_true):
    test_subjects = (
        pl.read_csv(
            "data/test_untransformed.csv",
            columns=[
                "src_subject_id",
                "eventname",
                "p_factor",
                "demo_sex_v2_1",
                "interview_age",
            ],
        )
        .with_columns(
            pl.col("eventname").replace(REVERSE_EVENT_MAPPING),
            pl.col("demo_sex_v2_1").replace(SEX_MAPPING),
            (pl.col("interview_age") / 12).round(0).cast(pl.Int32),
        )
        .with_columns(pl.all().forward_fill().over("src_subject_id"))
    )
    demographics = pl.read_csv(
        "data/demographics.csv",
        columns=["src_subject_id", "eventname", "race_ethnicity"],
    ).with_columns(
        pl.col("race_ethnicity").forward_fill().over("src_subject_id"),
        pl.col("eventname").replace(REVERSE_EVENT_MAPPING),
    )
    df = (
        test_subjects.join(demographics, on=["src_subject_id", "eventname"], how="left")
        .with_columns(y_pred=y_pred, y_true=y_true)
        .fill_null("Unknown")
    )
    print(df)
    df.write_csv("data/results/predicted_vs_observed.csv")


def make_shap_values(model, data_module):
    test_dataloader = iter(data_module.test_dataloader())
    X, _ = next(test_dataloader)
    background, _ = next(test_dataloader)
    explainer = GradientExplainer(model, background.to("mps:0"))
    shap_values = explainer.shap_values(X.to("mps:0"))
    save(shap_values, "data/results/shap_values.pt")


def regress_shap_values(dataloader):
    feature_names = (
        pl.read_csv("data/analytic/test.csv", n_rows=1)
        .drop(["src_subject_id", "p_factor"])
        .columns
    )
    test_dataloader = iter(dataloader)
    X, _ = next(test_dataloader)
    X = pd.DataFrame(X.view(-1, X.shape[2]), columns=feature_names)
    shap_values_list = torch_load("data/results/shap_values.pt")
    shap_values = np.mean(shap_values_list, axis=-1)
    shap_values = pd.DataFrame(
        shap_values.reshape(-1, shap_values.shape[2]), columns=feature_names
    )
    coefs = {name: [] for name in feature_names}
    n_bootstraps = 1000
    for _ in tqdm(range(n_bootstraps)):
        X_resampled, shap_resampled = resample(X, shap_values)
        for col in shap_values.columns:
            coef = (
                make_pipeline(StandardScaler(), LinearRegression())
                .fit(X_resampled[[col]], shap_resampled[[col]])
                .named_steps["linearregression"]
                .coef_[0, 0]
            )
            coefs[col].append(coef)
    df = pl.DataFrame(coefs).melt()
    df.write_csv("data/results/shap_coefs.csv")


def evaluate_model(config: Config, model, data_module):
    # make_shap_values(model, data_module)
    # regress_shap_values(dataloader=data_module.test_dataloader())
    y_pred, y_true = get_predictions(config, model, data_module)
    # r2_results(y_pred=y_pred, y_true=y_true)
    r2(y_pred=y_pred, y_true=y_true, config=config)
