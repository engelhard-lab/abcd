from functools import partial
from typing import Callable
import torch
import polars as pl
import polars.selectors as cs
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassSpecificityAtSensitivity,
    MulticlassSensitivityAtSpecificity,
)
from torchmetrics.wrappers import BootStrapper
from torchmetrics.functional import (
    roc,
    precision_recall_curve,
)
from multiprocessing import cpu_count

from abcd.config import Config
from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.model import Network, make_trainer
from abcd.preprocess import get_data

REVERSE_EVENT_MAPPING = {
    0: "Baseline",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
}
OUTPUT_COLUMNS = ["1", "2", "3", "4"]


def make_predictions(config: Config, model: Network):
    trainer, _ = make_trainer(config)
    _, _, test = get_data(config)
    data_module = ABCDDataModule(
        train=_,
        val=_,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=cpu_count(),
        dataset_class=RNNDataset,
    )
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
    outputs, _ = zip(*predictions)
    outputs = torch.concat(outputs).permute(0, 2, 1).contiguous().flatten(0, 1)
    metadata = pl.read_csv("data/test_metadata.csv")
    outputs = pl.DataFrame(outputs.cpu().numpy(), schema=OUTPUT_COLUMNS)
    return pl.concat([metadata, outputs], how="horizontal").drop_nulls("Next quartile")


def get_predictions(df: pl.DataFrame):
    outputs = torch.tensor(df.select(OUTPUT_COLUMNS).to_numpy()).float()
    labels = torch.tensor(df["Next quartile"].to_numpy()).long()
    return outputs, labels


def make_curve(df: pl.DataFrame, curve: Callable, name: str):
    outputs, labels = get_predictions(df)
    x, y, _ = curve(outputs, labels, task="multiclass", num_classes=outputs.shape[-1])
    if name == "PR":
        x, y = y, x
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "Metric": name,
                    "Curve": "Model",
                    "Variable": df["Variable"][0],
                    "Group": df["Group"][0],
                    "Next quartile": label,
                    "x": x_i.numpy(),
                    "y": y_i.numpy(),
                }
            )
            for label, x_i, y_i in zip(OUTPUT_COLUMNS, x, y)
        ]
    )


def bootstrap_metric(metric, outputs, labels, n_bootstraps=1000):
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=False, std=False, raw=True
    )
    bootstrap.update(outputs, labels)
    bootstraps = bootstrap.compute()["raw"].cpu().numpy()
    return pl.DataFrame(bootstraps, schema=OUTPUT_COLUMNS)


def make_metrics(df: pl.DataFrame):
    if df.shape[0] < 10:
        return pl.DataFrame(
            {
                "Metric": [],
                "Variable": [],
                "Group": [],
                "Next quartile": [],
                "value": [],
            }
        )
    outputs, labels = get_predictions(df)
    auroc = MulticlassAUROC(num_classes=outputs.shape[-1], average="none")
    ap = MulticlassAveragePrecision(num_classes=outputs.shape[-1], average="none")
    bootstrapped_auroc = bootstrap_metric(auroc, outputs, labels).with_columns(
        pl.lit("AUROC").alias("Metric")
    )
    bootstrapped_ap = bootstrap_metric(ap, outputs, labels).with_columns(
        pl.lit("AP").alias("Metric")
    )
    df = (
        pl.concat(
            [bootstrapped_auroc, bootstrapped_ap],
            how="diagonal_relaxed",
        )
        .with_columns(
            pl.lit(df["Group"][0]).cast(pl.String).alias("Group"),
            pl.lit(df["Variable"][0]).cast(pl.String).alias("Variable"),
        )
        .melt(id_vars=["Metric", "Variable", "Group"], variable_name="Next quartile")
    )
    return df


def make_prevalence(df: pl.DataFrame):
    df = (
        df.with_columns(pl.col("Next quartile").add(1))
        .to_dummies(["Next quartile"])
        .group_by("Variable", "Group")
        .agg(cs.contains("Next quartile_").mean())
        .rename(lambda x: x.replace("Next quartile_", ""))
        .melt(
            id_vars=["Variable", "Group"],
            variable_name="Next quartile",
            value_name="y",
        )
        .with_columns(
            pl.lit("PR").alias("Metric"),
            pl.lit("Baseline").alias("Curve"),
            pl.lit([0, 1]).alias("x"),
        )
    )
    # print(df)
    # df = df.with_columns(
    #     pl.col("Next quartile").count().over("Next quartile").alias("count")
    # )
    # df = df.with_columns(pl.count().over(["Variable", "Group"]).alias("n"))
    # print(
    #     df.filter(pl.col("Variable").eq("Quartile subset"))
    #     .group_by(["Variable", "Group"])
    #     .count()
    # )
    # df = df.with_columns(pl.col("n").truediv("count").alias("Prevalence"))
    # df = (
    #     df.group_by(["Next quartile", "Variable", "Group"])
    #     .count()
    #     # .with_columns(
    #     #     pl.col("count").truediv(pl.lit(n)).alias("y"),
    #     #     pl.col("Next quartile").add(1).cast(pl.String),
    #     #     pl.lit("PR").alias("Metric"),
    #     #     pl.lit("Baseline").alias("Curve"),
    #     #     pl.lit([0, 1]).alias("x"),
    #     # )
    #     # .drop("count")
    #     .drop_nulls()
    # )
    return df


def make_baselines(prevalence: pl.DataFrame):
    one_to_one_line = prevalence.with_columns(
        pl.lit("ROC").alias("Metric"),
        pl.lit("Baseline").alias("Curve"),
        pl.lit([0, 1]).alias("y"),
    ).explode("x", "y")
    prevalence = prevalence.explode("x")
    return pl.concat([prevalence, one_to_one_line], how="diagonal_relaxed")


def calc_sensitivity_and_specificity(df: pl.DataFrame):
    df = df.filter(pl.col("Quartile subset").eq("{1,2,3}"))
    outputs, labels = get_predictions(df=df)
    specificity = MulticlassSpecificityAtSensitivity(
        num_classes=outputs.shape[-1], min_sensitivity=0.5
    )
    spec = specificity(outputs, labels)
    sensitivity = MulticlassSensitivityAtSpecificity(
        num_classes=outputs.shape[-1], min_specificity=0.5
    )
    sens = sensitivity(outputs, labels)
    return spec, sens


def evaluate_model(config: Config, model: Network):
    if config.predict or not config.filepaths.results.predictions.is_file():
        df = make_predictions(config=config, model=model)
        df.write_csv(config.filepaths.results.predictions)
    else:
        df = pl.read_csv(config.filepaths.results.predictions)
    df = df.with_columns(
        pl.when(pl.col("Quartile").eq(3))
        .then(pl.lit("{4}"))
        .otherwise(pl.lit("{1,2,3}"))
        .alias("Quartile subset")
    )
    variables = [
        "Quartile subset",
        "Sex",
        "Race",
        "Age",
        "Year",
        "ADI quartile",
    ]
    df = df.melt(
        id_vars=OUTPUT_COLUMNS + ["Quartile", "Next quartile"],
        value_vars=variables,
        variable_name="Variable",
        value_name="Group",
    )
    df_all = df.filter(pl.col("Variable").eq("Quartile subset")).with_columns(
        pl.lit("{1,2,3,4}").alias("Group")
    )
    df = pl.concat([df_all, df]).drop_nulls("Group")
    # n = df_all.shape[0]
    prevalence = make_prevalence(df=df)
    grouped_df = df.group_by("Variable", "Group", maintain_order=True)
    metrics = grouped_df.map_groups(make_metrics)
    metrics = metrics.join(
        prevalence.select(["Variable", "Group", "Next quartile", "y"]),
        on=["Variable", "Group", "Next quartile"],
    ).rename({"y": "Prevalence"})
    metrics.write_csv("data/results/metrics.csv")
    pr_curve = grouped_df.map_groups(
        partial(make_curve, curve=precision_recall_curve, name="PR")
    )
    roc_curve = grouped_df.map_groups(partial(make_curve, curve=roc, name="ROC"))
    baselines = make_baselines(prevalence=prevalence)
    curves = pl.concat(
        [pr_curve, roc_curve, baselines],
        how="diagonal_relaxed",
    ).select(["Metric", "Curve", "Variable", "Group", "Next quartile", "x", "y"])
    curves.write_csv("data/results/curves.csv")

    spec, sens = calc_sensitivity_and_specificity(df=df)
    print(
        "Specificity:",
        spec[0].numpy().round(decimals=2),
        "threshold:",
        spec[1].numpy().round(decimals=2),
    )
    print(
        "Senstivity:",
        sens[0].numpy().round(decimals=2),
        "threshold:",
        sens[1].numpy().round(decimals=2),
    )
