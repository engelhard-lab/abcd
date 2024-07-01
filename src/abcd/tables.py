import polars as pl
import pandas as pd
import numpy as np


def quartile_counts_table():
    data = (
        pl.read_csv("data/metadata.csv")
        .rename({"Measurement year": "Year"})
        .with_columns(pl.col("Year").str.replace("Year ", ""))
        .to_pandas()
    )
    print(data)
    print(data["Age"].value_counts())
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


def make_metric_table(df: pl.DataFrame, groups: list[str]):
    df = (
        df.group_by(groups + ["Next quartile"], maintain_order=True)
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
        .sort(groups + ["Next quartile"])
        .pivot(values="value", columns="Next quartile", index=groups)
        .rename(
            {"1": "Quartile 1", "2": "Quartile 2", "3": "Quartile 3", "4": "Quartile 4"}
        )
    )
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


def make_tables():
    df = pl.read_csv("data/results/metrics.csv").with_columns(
        pl.col("Metric").cast(pl.Enum(["AUROC", "AP"])),
        pl.col("Group").str.replace("Year ", ""),
        pl.col("Variable").str.replace("Measurement year", "Year"),
    )
    metrics_df = df.filter(pl.col("Variable").eq("Quartile subset"))
    metrics_table = make_metric_table(df=metrics_df, groups=["Metric", "Group"])
    print(metrics_table)
    metrics_table.write_csv("data/results/tables/metrics.csv")
    demographic_df = df.filter(pl.col("Variable").ne("Quartile subset"))
    demographic_metrics = make_metric_table(
        df=demographic_df, groups=["Metric", "Variable", "Group"]
    ).sort(
        [
            "Metric",
            "Variable",
            pl.col("Group").cast(pl.Int32, strict=False),
        ]
    )
    demographic_metrics.write_csv("data/results/tables/demographic_metrics.csv")
    print(demographic_metrics)
    # print(df)
    # shap_table()
    # make_follow_up_table()
    # make_analysis_demographics()
    # quartile_counts_table()


if __name__ == "__main__":
    make_tables()
