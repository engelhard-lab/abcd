from multiprocessing import cpu_count
from typing import Callable
import torch
import polars as pl
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassAveragePrecision,
)
from torchmetrics.wrappers import BootStrapper
from torchmetrics.functional import (
    roc,
    precision_recall_curve,
)

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


def bootstrap_metric(metric, outputs, labels, n_bootstraps=1000):
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=False, std=False, raw=True
    )
    bootstrap.update(outputs, labels)
    bootstrapped = bootstrap.compute()["raw"]
    columns = [f"Q{i}" for i in range(1, 5)]
    return pl.DataFrame(bootstrapped.numpy(), schema=columns)


def make_metrics(outputs, labels):
    metrics = {
        "AUROC": MulticlassAUROC(num_classes=outputs.shape[-1], average="none"),
        "Accuracy": MulticlassAccuracy(num_classes=outputs.shape[-1], average="none"),
        "Average Precision": MulticlassAveragePrecision(
            num_classes=outputs.shape[-1], average="none"
        ),
        # MulticlassCalibrationError(num_classes=outputs.shape[-1]),
    }
    dfs = []
    for name, metric in metrics.items():
        df = bootstrap_metric(metric, outputs, labels).with_columns(
            pl.lit(name).alias("Metric")
        )
        dfs.append(df)
    return pl.concat(dfs)


def make_metrics_subsets(predictions: dict):
    dfs = []
    for (q4_t, q4_tp1), (outputs, labels) in predictions.items():
        df = make_metrics(outputs, labels).with_columns(
            pl.lit(q4_t).alias("Quartile$_t$"),
            pl.lit(q4_tp1).alias("Quartile$_{t+1}$"),
        )
        dfs.append(df)
    return pl.concat(dfs)


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
            "p-factor quartile$_{t+1}$": i + 1,
        }
        for i, (precision_i, recall_i) in enumerate(zip(precision, recall))
    ]
    return pl.DataFrame(data).explode(columns=["y", "x"])


def make_curve_groups(predictions, metric: Callable):
    dfs = []
    for (q4_t, q4_tp1), (outputs, labels) in predictions.items():
        df = metric(outputs, labels).with_columns(
            pl.lit(q4_t).alias("Quartile$_t$"),
            pl.lit(q4_tp1).alias("Quartile$_{t+1}$"),
        )
        dfs.append(df)
    return pl.concat(dfs).filter(
        pl.col("p-factor quartile$_{t+1}$").eq(4)
        | pl.col("Qartile$_t$").eq("$\\{1, 2, 3, 4\\}$")
    )


def make_metric_curves(predictions):
    metrics = {"ROC": roc_curve, "PR": pr_curve}
    dfs = []
    for name, metric in metrics.items():
        df = make_curve_groups(predictions=predictions, metric=metric).with_columns(
            pl.lit(name).alias("Metric")
        )
        dfs.append(df)
    return pl.concat(dfs)


def evaluate_model(config: Config, model: Network):
    outputs, labels, quartiles = make_predictions(config=config, model=model)
    outputs_q4_t = outputs[quartiles == 3]
    labels_q4_t = labels[quartiles == 3]
    outputs_not_q4_t = outputs[quartiles != 3]
    labels_not_q4_t = labels[quartiles != 3]
    predictions = {
        ("$\\{1, 2, 3, 4\\}$", "$\\{1, 2, 3, 4\\}$"): (outputs, labels),
        ("$\\{4\\}$", "$\\{4\\}$"): (outputs_q4_t, labels_q4_t),
        ("$\\{1, 2, 3\\}$", "$\\{4\\}$"): (outputs_not_q4_t, labels_not_q4_t),
    }
    metrics = make_metrics_subsets(predictions)
    metrics.write_csv("data/results/metrics.csv")
    curves = make_metric_curves(predictions)
    curves.write_csv("data/results/curves.csv")
