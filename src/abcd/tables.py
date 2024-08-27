import polars as pl
import pandas as pd
import numpy as np


def quartile_counts():
    data = (
        pl.read_csv("data/metadata.csv")
        .rename({"Measurement year": "Year"})
        .with_columns(pl.col("Year").str.replace("Year ", ""))
        .to_pandas()
    )
    data["Next quartile"] = data["Next quartile"] + 1
    quartile = data["Next quartile"]
    variables = ["Sex", "Age", "Race", "Year", "ADI quartile"]
    dfs = []
    for i, name in enumerate(variables):
        variable = data[name]
        df1 = pd.crosstab(
            [variable],
            quartile,
            rownames=["Group"],
            colnames=["Next quartile"],
            margins=True,
        ).astype(str)
        df2 = " (" + (
            pd.crosstab(
                [variable],
                quartile,
                rownames=["Group"],
                colnames=["Next quartile"],
                margins=True,
                normalize="all",
            )
            .mul(100)
            .round(1)
            .astype(str)
            + "%)"
        )
        df = df1 + df2
        df = df.reset_index()
        df["Group"] = [int(i) if isinstance(i, float) else i for i in df["Group"]]
        df.insert(0, "Variable", name)
        if i == 0:
            df = df.reindex(np.roll(df.index, shift=1))
            df.iloc[0, 0] = "All"
        else:
            df = df.iloc[:-1]
        dfs.append(df)
    df = pd.concat(dfs).rename(
        columns={
            1: "Quartile 1",
            2: "Quartile 2",
            3: "Quartile 3",
            4: "Quartile 4",
            "All": "Total",
        }
    )
    print(df)
    df.to_csv("data/results/tables/quartile_counts.csv", index=False)


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
        .write_csv("data/tables/demographic_counts.csv")
    )


def make_metric_table(df: pl.DataFrame, groups: list[str]):
    df = (
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
        .sort(groups + ["Quartile at t+1"])
        .pivot(values="value", columns="Quartile at t+1", index=groups)
        .rename(
            {"1": "Quartile 1", "2": "Quartile 2", "3": "Quartile 3", "4": "Quartile 4"}
        )
    )
    print(df)
    return df


def shap_table():
    df = pl.read_csv("data/results/shap_coefs.csv")
    variables = pl.read_csv("data/variables.csv").rename({"column": "variable"})
    df = df.group_by("variable").agg(
        pl.col("value").mean().alias("shap_coef_mean"),
        pl.col("value").std().alias("shap_coef_std"),
    )
    df = (
        df.join(variables, on="variable", how="left")
        .select(
            [
                "dataset",
                "table",
                "respondent",
                "variable",
                "shap_coef_mean",
                "shap_coef_std",
                "question",
                "response",
            ]
        )
        .sort(["dataset", "table", "respondent", "variable"])
    )
    df.write_csv("data/results/tables/shap_coefs.csv")


def format_analysis(df, name):
    return df.with_columns(pl.lit(name).alias("Analysis")).with_columns(
        pl.col("Analysis").str.replace("_", " + ").str.to_titlecase()
    )


def aggregate_metrics():
    names = [
        # "by_year",
        # "binary",
        "questions",
        "questions_brain",
        "autoregressive",
        "questions_autoregressive",
        "questions_symptoms",
        "symptoms",
        "all",
    ]
    metrics = []
    curves = []
    for name in names:
        metric = pl.read_csv(f"data/analyses/{name}/results/metrics/metrics.csv").pipe(
            format_analysis, name=name
        )
        metrics.append(metric)
        curve = pl.read_csv(f"data/analyses/{name}/results/metrics/curves.csv").pipe(
            format_analysis, name=name
        )
        curves.append(curve)
    metrics = pl.concat(metrics)
    curves = pl.concat(curves)
    metrics.write_csv("data/tables/analyses_metrics.csv")
    curves.write_csv("data/tables/analyses_curves.csv")


def make_tables():
    demographic_counts()
    # aggregate_metrics()
    # df = pl.read_csv("data/tables/analyses_metrics.csv")
    # metrics_df = df.filter(pl.col("Variable").eq("Quartile subset")).with_columns(
    #     pl.col("Metric").cast(pl.Enum(["AUROC", "AP"])),
    #     # pl.col("Group").str.replace("Year ", ""),
    #     # pl.col("Variable").str.replace("Measurement year", "Year"),
    # )
    # metrics_table = make_metric_table(
    #     df=metrics_df, groups=["analysis", "Metric", "Group"]
    # )
    # metrics_table.write_csv("data/results/tables/quartile_subset_summary.csv")

    # demographic_df = df.filter(pl.col("Variable").ne("Quartile subset"))
    # demographic_metrics = make_metric_table(
    #     df=demographic_df, groups=["Metric", "Variable", "Group"]
    # ).sort(
    #     [
    #         "Metric",
    #         "Variable",
    #         pl.col("Group").cast(pl.Int32, strict=False),
    #     ]
    # )
    # demographic_metrics.write_csv("data/results/tables/demographic_metrics.csv")
    # print(demographic_metrics)
    # print(df)
    # shap_table()
    # make_follow_up_table()
    # make_analysis_demographics()
    # quartile_counts()


if __name__ == "__main__":
    make_tables()
