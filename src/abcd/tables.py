import polars as pl
import pandas as pd
import numpy as np


# def make_follow_up_table():
#     df = pl.read_csv("data/analytic/dataset.csv").with_columns(
#         pl.col("eventname").replace(EVENT_MAPPING_2)
#     )
#     labels = pl.read_csv("data/labels/factor_scores.csv").with_columns(
#         pl.col("eventname").replace(EVENT_MAPPING_1)
#     )
#     features = pl.read_csv("data/analytic/features.csv").with_columns(
#         pl.col("eventname").replace(EVENT_MAPPING_1)
#     )
#     df = (
#         df.group_by("eventname")
#         .len()
#         .sort("eventname")
#         .with_columns(pl.col("eventname").cast(pl.Utf8))
#         .with_columns(pl.lit("Joined").alias("Dataset"))
#     )
#     labels = (
#         labels.group_by("eventname")
#         .len()
#         .sort("eventname")
#         .with_columns(pl.lit("p-factor").alias("Dataset"))
#     )
#     features = (
#         features.group_by("eventname")
#         .len()
#         .sort("eventname")
#         .with_columns(pl.lit("Predictors").alias("Dataset"))
#     )
#     df = (
#         pl.concat([df, labels, features])
#         .pivot(values="len", columns="Dataset", index="eventname")
#         .rename({"eventname": "Year"})
#     )
#     df.select("Year", "p-factor", "Predictors", "Joined").drop_nulls().write_csv(
#         "data/results/tables/follow_up.csv"
#     )


# def make_p_factor_group(df: pl.DataFrame, group_name: str, variable_name: str):
#     return (
#         df.group_by(group_name)
#         .agg(
#             (
#                 pl.col("p_factor").mean().round(3).cast(pl.Utf8)
#                 + " ± "
#                 + pl.col("p_factor").std().mul(2).round(2).cast(pl.Utf8)
#             ).alias("p-factor"),
#             pl.col("src_subject_id").n_unique().alias("Count"),
#         )
#         .sort(group_name)
#         .rename({group_name: "Group"})
#         .with_columns(
#             pl.lit(variable_name).alias("Variable"), pl.col("Group").cast(pl.Utf8)
#         )
#     )


def quartile_counts_table():
    data = pl.read_csv("data/metadata.csv").to_pandas()
    data["Next quartile"] = data["Next quartile"] + 1
    quartile = data["Next quartile"]
    variables = ["Sex", "Age", "Race", "Year"]
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
        (
            (
                df.group_by(groups + ["Next quartile"]).agg(
                    pl.col("Prevalence").first(),
                    pl.col("value").mean().round(2).cast(pl.String)
                    + " ± "
                    + pl.col("value").std(2).round(2).cast(pl.String),
                )
            )
            .with_columns(pl.col("Prevalence").round(2))
            .with_columns(
                pl.when(pl.col("Metric").eq("AP"))
                .then(
                    pl.col("value").add(" (" + pl.col("Prevalence").cast(pl.String))
                    + ")"
                )
                .otherwise(pl.col("value"))
            )
        )
        .sort(groups + ["Next quartile"])
        .pivot(values="value", columns="Next quartile", index=groups)
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
    df = pl.read_csv("data/results/metrics.csv")
    # print(df)
    df = make_metric_table(df, groups=["Metric", "Group"])
    print(df)
    # df.write_csv("data/results/tables/metrics.csv")
    # print(df)
    # df = pl.read_csv("data/results/demographic_metrics.csv")
    # df = make_metric_table(df, groups=["Metric", "Variable", "Group"])
    # df.write_csv("data/results/tables/demographic_metrics.csv")
    # print(df)
    # shap_table()
    # make_follow_up_table()
    # make_analysis_demographics()
    # quartile_counts_table()


if __name__ == "__main__":
    make_tables()
