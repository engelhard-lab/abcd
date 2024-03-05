from sklearn.metrics import r2_score
from abcd.evaluate import RACE_MAPPING, SEX_MAPPING
from abcd.preprocess import EVENT_MAPPING
import polars as pl


def make_follow_up_table():
    df = pl.read_csv("data/analytic/dataset.csv")
    labels = pl.read_csv("data/labels/factor_scores.csv").with_columns(
        pl.col("eventname").replace(EVENT_MAPPING)
    )
    features = pl.read_csv("data/analytic/features.csv").with_columns(
        pl.col("eventname").replace(EVENT_MAPPING)
    )
    df = (
        df.group_by("eventname")
        .len()
        .sort("eventname")
        .with_columns(pl.col("eventname").cast(pl.Utf8))
        .with_columns(pl.lit("Joined").alias("Dataset"))
    )
    labels = (
        labels.group_by("eventname")
        .len()
        .sort("eventname")
        .with_columns(pl.lit("p-factor").alias("Dataset"))
    )
    features = (
        features.group_by("eventname")
        .len()
        .sort("eventname")
        .with_columns(pl.lit("Predictors").alias("Dataset"))
    )
    df = (
        pl.concat([df, labels, features])
        .pivot(values="len", columns="Dataset", index="eventname")
        .rename({"eventname": "Year"})
    )
    df.select("Year", "p-factor", "Predictors", "Joined").write_csv(
        "data/results/tables/follow_up.csv"
    )


# FIXME add correlation, add counts
def p_factor_table():
    df = pl.read_csv("data/labels/factor_scores.csv", null_values="NA")
    demographics = pl.read_csv("data/demographics.csv").with_columns(
        pl.col("eventname")
    )
    df = (
        df.join(demographics, on=["src_subject_id", "eventname"], how="inner")
        .drop_nulls()
        .with_columns(
            pl.col("eventname").replace(EVENT_MAPPING),
            pl.col("race_ethnicity").replace(RACE_MAPPING),
            pl.col("sex").replace(SEX_MAPPING),
        )
        .filter(pl.col("sex") != "3")
    )
    year = (
        df.group_by("eventname")
        .agg(
            (
                pl.col("p_score").mean().round(3).cast(pl.Utf8)
                + " ± "
                + pl.col("p_score").std().mul(2).round(2).cast(pl.Utf8)
            ).alias("Mean p-factor ± 2std"),
        )
        .sort("eventname")
        .rename({"eventname": "Group"})
        .with_columns(pl.lit("Year").alias("Variable"), pl.col("Group").cast(pl.Utf8))
    )
    sex = (
        df.group_by("sex")
        .agg(
            # pl.col("p_score").count().alias("count"),
            (
                pl.col("p_score").mean().round(3).cast(pl.Utf8)
                + " ± "
                + pl.col("p_score").std().mul(2).round(2).cast(pl.Utf8)
            ).alias("Mean p-factor ± 2std"),
        )
        .sort("sex")
        .rename({"sex": "Group"})
        .with_columns(pl.lit("Sex").alias("Variable"))
    )
    # counts = df.select(pl.col("race_ethnicity").value_counts().alias("count"))
    # print(counts)
    race = (
        df.group_by("race_ethnicity")
        .agg(
            (
                pl.col("p_score").mean().round(3).cast(pl.Utf8)
                + " ± "
                + pl.col("p_score").std().mul(2).round(2).cast(pl.Utf8)
            ).alias("Mean p-factor ± 2std"),
        )
        .sort("race_ethnicity")
        .rename({"race_ethnicity": "Group"})
        .with_columns(pl.lit("Race/Ethnicity").alias("Variable"))
    )
    age = (
        df.with_columns((pl.col("age") / 12).round(0).cast(pl.Int32).cast(pl.Utf8))
        .group_by("age")
        .agg(
            (
                pl.col("p_score").mean().round(3).cast(pl.Utf8)
                + " ± "
                + pl.col("p_score").std().mul(2).round(2).cast(pl.Utf8)
            ).alias("Mean p-factor ± 2std"),
        )
        .sort("age")
        .rename({"age": "Group"})
        .with_columns(pl.lit("Age").alias("Variable"), pl.col("Group").cast(pl.Utf8))
    )
    df = pl.concat([sex, year, race, age]).select(
        "Variable", "Group", "Mean p-factor ± 2std"
    )
    print(df)
    df.write_csv("data/results/tables/p_factors.csv")


def apply_r2(args) -> pl.Series:
    return pl.Series([r2_score(y_true=args[0], y_pred=args[1])], dtype=pl.Float32)


def r2_table():
    df = pl.read_csv("data/results/predicted_vs_observed.csv").with_columns(
        # pl.col("eventname").replace(),
        pl.col("race_ethnicity").replace(RACE_MAPPING),
        pl.col("sex").replace(SEX_MAPPING),
    )
    year = (
        df.group_by("eventname")
        .agg(pl.map_groups(exprs=["y_true", "y_pred"], function=apply_r2).alias("r2"))
        .rename({"eventname": "Group"})
        .with_columns(pl.lit("Year").alias("Variable"), pl.col("Group").cast(pl.Utf8))
        .sort("Group")
    )
    sex = (
        df.group_by("sex")
        .agg(pl.map_groups(exprs=["y_true", "y_pred"], function=apply_r2).alias("r2"))
        .rename({"sex": "Group"})
        .with_columns(pl.lit("Sex").alias("Variable"))
        .sort("Group")
    )
    race = (
        df.group_by("race_ethnicity")
        .agg(pl.map_groups(exprs=["y_true", "y_pred"], function=apply_r2).alias("r2"))
        .rename({"race_ethnicity": "Group"})
        .with_columns(pl.lit("Race/Ethnicity").alias("Variable"))
        .sort("Group")
    )
    age = (
        df.with_columns((pl.col("age") / 12).round(0).cast(pl.Int32).cast(pl.Utf8))
        .group_by("age")
        .agg(pl.map_groups(exprs=["y_true", "y_pred"], function=apply_r2).alias("r2"))
        .rename({"age": "Group"})
        .with_columns(pl.lit("Age").alias("Variable"))
        .sort("Group")
        .filter(pl.col("r2") > 0)
    )
    df = (
        pl.concat([year, sex, race, age])
        .select("Variable", "Group", "r2")
        .with_columns(pl.col("r2").round(2))
    )
    df.write_csv("data/results/r2_by_event.csv")


if __name__ == "__main__":
    # make_follow_up_table()
    p_factor_table()
    # r2_table()
