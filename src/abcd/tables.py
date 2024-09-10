from itertools import product
import polars as pl
from tqdm import tqdm

from abcd.config import Config, get_config

group_order = pl.Enum(
    [
        "1",
        "2",
        "3",
        "4",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "Baseline",
        "1-year",
        "2-year",
        "3-year",
        "Asian",
        "Black",
        "Hispanic",
        "White",
        "Other",
        "Female",
        "Male",
    ]
)


def cross_tabulation():
    df = pl.read_csv("data/raw/metadata.csv")
    risk_groups = {
        "1": "No risk",
        "2": "Low risk",
        "3": "Moderate risk",
        "4": "High risk",
    }
    variables = ["Sex", "Age", "Race", "Follow-up event", "Event year", "ADI quartile"]
    columns = ["Variable", "Group", "No risk", "Low risk", "Moderate risk", "High risk"]
    df = (
        df.select(["Quartile at t+1"] + variables)
        .with_columns(pl.col("Quartile at t+1").replace_strict(risk_groups))
        .unpivot(index="Quartile at t+1", on=variables)
        .group_by("Quartile at t+1", "variable", "value")
        .len()
        .pivot(index=["variable", "value"], on="Quartile at t+1", values="len")
        .rename({"variable": "Variable", "value": "Group"})
        .select(columns)
        .sort(
            "Variable",
            pl.col("Group").cast(
                pl.Enum(["Baseline", "1-year", "2-year", "3-year"]), strict=False
            ),
        )
        .with_columns(pl.sum_horizontal(pl.exclude("Variable", "Group")).alias("Total"))
    )
    col_sum = pl.exclude("Variable", "Group").sum().over("Variable")
    percent = (
        pl.exclude("Variable", "Group")
        .truediv(col_sum)
        .mul(100)
        .round(0)
        .cast(pl.Int32)
        .cast(pl.String)
    )
    df = (
        df.with_columns(
            pl.exclude("Variable", "Group").cast(pl.String) + " (" + percent + "%)"
        )
        .drop_nulls()
        .sort("Variable", pl.col("Group").cast(group_order, strict=False))
    )
    df.write_csv("data/tables/table_1.csv")


def demographic_counts():
    columns = ["Subject ID", "Sex", "Race", "ADI quartile"]
    df = pl.read_csv("data/raw/metadata.csv").select(columns).unique(columns)
    n = df["Subject ID"].n_unique()
    df = (
        df.melt(id_vars="Subject ID")
        .group_by("variable")
        .agg(pl.col("value").value_counts())
        .explode("value")
        .unnest("value")
        .with_columns(
            pl.col("count")
            .truediv(n)
            .mul(100)
            .round(1)
            .cast(pl.String)
            .add("%")
            .alias("percentage")
        )
        .sort("variable", "value")
        .write_csv("data/supplement/tables/supplemental_table_1.csv")
    )


def shap_table(analysis: str, factor_model: str):
    columns = [
        "dataset",
        "table",
        "respondent",
        "question",
        "response",
    ]
    df = pl.read_csv(
        f"data/analyses/{factor_model}/{analysis}/results/shap_values.csv"
    ).rename({"Respondent": "respondent"})
    df = (
        df.group_by("variable")
        .agg(
            pl.col(columns).first(),
            pl.col("shap_value").sum().alias("shap_value"),
        )
        .sort(pl.col("shap_value").abs(), descending=True)
    ).select(["variable"] + columns + ["shap_value"])
    df.write_csv("data/supplement/tables/supplemental_table_4.csv")


def aggregate_metrics(analyses: list[str], factor_models: list[str]):
    for metric_type in ("sensitivity_specificity", "curves", "metrics"):
        metrics = []
        progress_bar = tqdm(
            product(analyses, factor_models), total=len(analyses) * len(factor_models)
        )
        for analysis, factor_model in progress_bar:
            path = f"data/analyses/{factor_model}/{analysis}/results/metrics/{metric_type}.csv"
            metric = pl.scan_csv(path).with_columns(
                pl.lit(factor_model).alias("Factor model"),
                pl.lit(analysis).alias("Predictor set"),
            )
            metrics.append(metric)
        pl.concat(metrics).with_columns(
            pl.col("Predictor set").replace(
                {
                    "questions": "Questionnaires",
                    "symptoms": "CBCL scales",
                    "questions_symptoms": "Questionnaires, CBCL scales",
                    "questions_mri": "Questionnaires, MRI",
                    "questions_mri_symptoms": "Questionnaires, MRI, CBCL scales",
                    "autoregressive": "Previous p-factors",
                }
            ),
            pl.col("Factor model").replace(
                {"within_event": "Within-event", "across_event": "Across-event"}
            ),
        ).sink_parquet(f"data/results/metrics/{metric_type}.parquet")


def make_metric_table(df: pl.LazyFrame, groups: list[str]):
    group_order = [
        pl.col("Group").cast(pl.Int32, strict=False) if group == "Group" else group
        for group in groups
    ]
    return (
        df.group_by(groups + ["Quartile at t+1"], maintain_order=True)
        .agg(
            pl.col("Prevalence").first(),
            pl.col("value").mean().round(2).cast(pl.String)
            + " ± "
            + pl.col("value").std(2).round(2).cast(pl.String),
        )
        .with_columns(pl.col("Prevalence").round(2))
        .with_columns(
            pl.when(pl.col("Metric").eq("AP"))
            .then(
                pl.col("value").add(" (" + pl.col("Prevalence").cast(pl.String)) + ")"
            )
            .otherwise(pl.col("value"))
        )
        .sort("Quartile at t+1")
        .collect()
        .pivot(on="Quartile at t+1", values="value", index=groups)
        .rename(
            {"1": "No risk", "2": "Low risk", "3": "Moderate risk", "4": "High risk"}
        )
        .with_columns(pl.col("Metric").cast(pl.Enum(["AUROC", "AP"])))
        .sort(group_order)
        .lazy()
        .sink_parquet("data/results/metrics/metric_summary.parquet")
    )


def quartile_metric_table(df: pl.LazyFrame):
    return (
        df.filter(
            pl.col("Variable").eq("High-risk scenario")
            & pl.col("Predictor set").is_in(["Questionnaires", "CBCL scales"])
            & pl.col("Factor model").eq("Within-event")
        )
        .with_columns(
            pl.col("Group").cast(pl.Enum(["Conversion", "Persistence", "Agnostic"])),
            pl.col("Predictor set").cast(pl.Enum(["Questionnaires", "CBCL scales"])),
        )
        .drop("Factor model", "Variable")
        .sort("Predictor set", "Metric", "Group")
        .rename({"Group": "High-risk scenario"})
    )


def demographic_metric_table(df: pl.LazyFrame):
    return (
        df.filter(
            pl.col("Variable").ne("High-risk scenario")
            & pl.col("Predictor set").eq("Questionnaires")
            & pl.col("Factor model").eq("Within-event")
        )
        .drop("Factor model", "Predictor set")
        .sort("Metric", "Variable", pl.col("Group").cast(group_order))
    )


def make_tables(config: Config):
    cross_tabulation()
    demographic_counts()
    aggregate_metrics(analyses=config.analyses, factor_models=config.factor_models)
    df = pl.scan_parquet("data/results/metrics/metrics.parquet")
    groups = ["Factor model", "Predictor set", "Metric", "Variable", "Group"]
    make_metric_table(df=df, groups=groups)
    metric_table = pl.scan_parquet("data/results/metrics/metric_summary.parquet")
    metric_table.sink_csv("data/supplement/tables/supplemental_table_2.csv")
    quartile_metrics = quartile_metric_table(df=metric_table)
    quartile_metrics.collect().write_csv("data/tables/table_2.csv")
    demographic_metrics = demographic_metric_table(df=metric_table)
    demographic_metrics.collect().write_csv("data/tables/table_3.csv")
    shap_table(analysis="questions", factor_model="within_event")


if __name__ == "__main__":
    config = get_config()
    make_tables(config)
