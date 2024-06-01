from multiprocessing import cpu_count
from operator import not_
from typing import Callable
from shap import GradientExplainer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import torch
import torch.nn.functional as F
import polars as pl
from tqdm import tqdm
from torchmetrics.functional import (
    auroc,
    average_precision,
    roc,
    precision_recall_curve,
)
import pandas as pd
import numpy as np

from abcd.config import Config
from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.model import Network, make_trainer
from abcd.preprocess import get_data
from abcd.tables import SEX_MAPPING

REVERSE_EVENT_MAPPING = {
    0: "Baseline",
    1: "Year 1",
    2: "Year 2",
    3: "Year 3",
    4: "Year 4",
}


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


def auc_bootstrap(outputs, labels):
    aucs = []
    n_bootstraps = 1000
    for _ in tqdm(range(n_bootstraps)):
        outputs_resampled: torch.Tensor
        labels_resampled: torch.Tensor
        outputs_resampled, labels_resampled = resample(outputs, labels)  # type: ignore
        auc = auroc(
            outputs_resampled,
            labels_resampled,
            task="multiclass",
            num_classes=outputs.shape[-1],
            average="none",
        )  # type: ignore
        auc = {f"Q{i+1}": value.item() for i, value in enumerate(auc)}  # type: ignore
        aucs.append(auc)
    return pl.DataFrame(aucs)


def make_predictions(config: Config, model):
    trainer, _ = make_trainer(config)
    train, val, test = get_data(config)
    data_module = ABCDDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=cpu_count(),
        dataset_class=RNNDataset,
    )
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
    outputs, labels, quartiles = zip(*predictions)
    outputs = torch.concat(outputs).permute(0, 2, 1).contiguous().flatten(0, 1)
    labels = torch.concat(labels).long().flatten()
    quartiles = torch.concat(quartiles).long().flatten()
    return outputs, labels, quartiles


def roc_curve(outputs: torch.Tensor, labels: torch.Tensor):
    fpr, tpr, _ = roc(outputs, labels, task="multiclass", num_classes=outputs.shape[-1])
    data = [
        {
            "y": tpr_i.numpy().tolist(),
            "x": fpr_i.numpy().tolist(),
            "p-factor quartile$_{t+1}$": i,
        }
        for i, (tpr_i, fpr_i) in enumerate(zip(tpr, fpr))
    ]
    return pl.DataFrame(data).explode(columns=["y", "x"])


def pr_curve(outputs: torch.Tensor, labels: torch.Tensor):
    precision, recall, _ = precision_recall_curve(
        outputs, labels, task="multiclass", num_classes=outputs.shape[-1]
    )
    data = [
        {
            "y": precision_i.numpy().tolist(),
            "x": recall_i.numpy().tolist(),
            "p-factor quartile$_{t+1}$": i,
        }
        for i, (precision_i, recall_i) in enumerate(zip(precision, recall))
    ]
    return pl.DataFrame(data).explode(columns=["y", "x"])


def metric_groups(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    quartiles: torch.Tensor,
    metric: Callable,
):
    df = metric(outputs, labels).with_columns(
        pl.lit("$\\{1, 2, 3, 4\\}$").alias("p-factor quartile$_t$"),
        pl.lit("p-factor quartile$_{t+1} \\in \\{1, 2, 3, 4\\}$").alias("group"),
    )
    in_q4 = metric(outputs[quartiles == 3], labels[quartiles == 3]).with_columns(
        pl.lit("$\\{4\\}$").alias("p-factor quartile$_t$"),
        pl.lit("p-factor quartile$_{t+1} \\in \\{4\\}$").alias("group"),
    )
    not_in_q4 = metric(
        outputs=outputs[quartiles != 3], labels=labels[quartiles != 3]
    ).with_columns(
        pl.lit("$\\{1, 2, 3\\}$").alias("p-factor quartile$_t$"),
        pl.lit("p-factor quartile$_{t+1} \\in \\{4\\}$").alias("group"),
    )
    return (
        pl.concat([df, in_q4, not_in_q4])
        .filter(
            pl.col("group").eq("p-factor quartile$_{t+1} \\in \\{1, 2, 3, 4\\}$")
            | (
                pl.col("group").eq("p-factor quartile$_{t+1} \\in \\{4\\}$")
                & pl.col("p-factor quartile$_{t+1}$").eq(3)
            )
        )
        .with_columns(pl.col("p-factor quartile$_{t+1}$").add(1))
    )


def evaluate_model(config: Config, model: Network):
    outputs, labels, quartiles = make_predictions(config=config, model=model)
    measurement = torch.arange(1, 5).repeat(
        outputs.shape[0] // 4
    )  # TODO make this dynamic based on number of measurements
    (
        pl.DataFrame(
            {
                "output": outputs.numpy(),
                "quartile_tp1": labels.numpy(),
                "quartile_t": quartiles.numpy(),
                "measurement": measurement.numpy(),
            }
        )
        .with_columns(pl.col("output").list.to_struct(fields=lambda idx: f"Q{idx+1}"))
        .unnest("output")
    ).write_csv("data/results/predictions.csv")
    roc_df = metric_groups(outputs, labels, quartiles, roc_curve).with_columns(
        pl.lit("ROC").alias("metric")
    )
    pr_df = metric_groups(outputs, labels, quartiles, pr_curve).with_columns(
        pl.lit("PR").alias("metric")
    )
    df = pl.concat([roc_df, pr_df])
    print(df)
    df.write_csv("data/results/roc_pr.csv")
