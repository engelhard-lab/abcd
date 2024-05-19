from multiprocessing import cpu_count
from shap import GradientExplainer
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch import concat, save, load as torch_load
import polars as pl
from tqdm import tqdm
from torchmetrics.functional import (
    auroc,
    average_precision,
)
import pandas as pd
import numpy as np

from abcd.config import Config
from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.model import make_trainer
from abcd.preprocess import get_data
from abcd.tables import SEX_MAPPING

REVERSE_EVENT_MAPPING = {
    0: "Baseline",
    1: "Year 1",
    2: "Year 2",
    3: "Year 3",
    4: "Year 4",
}


# FIXME
def get_predictions(trainer, model, data_module):
    predictions = trainer.predict(
        model=model, dataloaders=data_module.test_dataloader()
    )
    outputs, labels = zip(*predictions)
    outputs = concat(outputs)  # .nanmean(dim=1)
    labels = concat(labels)  # .nanmean(dim=1)
    return outputs, labels


def r2(outputs, labels, config: Config):
    r2s = []
    for _ in range(1000):
        outputs_resampled, labels_resampled = resample(outputs, labels)
        r2 = r2_score(
            y_pred=labels_resampled, y_true=outputs_resampled, multioutput="raw_values"
        )
        r2s.append(r2)
    r2 = pl.DataFrame(r2s, schema=config.labels.cbcl_labels).melt()
    r2.write_csv(f"data/results/r2_{config.target}.csv")


def make_metadata():
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
    return test_subjects.join(
        demographics, on=["src_subject_id", "eventname"], how="left"
    ).fill_null("Unknown")


# FIXME
def evaluate_model(targets, config: Config, model, data_module):
    trainer, _ = make_trainer(config)
    train, val, test = get_data(config)
    test = (
        pl.DataFrame(test)
        .filter(pl.col("p_factor").eq(3) & pl.col("p_factor").shift(1).eq(3).is_not())
        .to_pandas()
    )
    data_module = ABCDDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=cpu_count(),
        dataset_class=RNNDataset,
        target=config.target,
    )
    outputs, labels = get_predictions(trainer, model, data_module)
    auroc_score = auroc(
        outputs, labels.long(), task="multiclass", num_classes=outputs.shape[-1]
    )
    ap_score = average_precision(
        outputs, labels.long(), task="multiclass", num_classes=outputs.shape[-1]
    )
    print(auroc_score, ap_score)
    # metadata = make_metadata()
    # metadata.write_csv("data/results/predicted_vs_observed.csv")
