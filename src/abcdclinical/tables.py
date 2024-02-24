from abcdclinical.dataset import EVENT_MAPPING
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


if __name__ == "__main__":
    make_follow_up_table()
