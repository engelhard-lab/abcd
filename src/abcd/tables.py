from sklearn.metrics import r2_score
import polars as pl

RACE_MAPPING = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
SEX_MAPPING = {1: "Male", 0: "Female"}
EVENT_MAPPING_1 = {
    "baseline_year_1_arm_1": "0",
    "1_year_follow_up_y_arm_1": "1",
    "2_year_follow_up_y_arm_1": "2",
    "3_year_follow_up_y_arm_1": "3",
    "4_year_follow_up_y_arm_1": "4",
}
EVENT_MAPPING_2 = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
COLUMNS = [
    ("eventname", "Year"),
    ("demo_sex_v2_1", "Sex"),
    ("race_ethnicity", "Race/Ethnicity"),
    ("interview_age", "Age"),
]


def make_follow_up_table():
    df = pl.read_csv("data/analytic/dataset.csv").with_columns(
        pl.col("eventname").replace(EVENT_MAPPING_2)
    )
    labels = pl.read_csv("data/labels/factor_scores.csv").with_columns(
        pl.col("eventname").replace(EVENT_MAPPING_1)
    )
    features = pl.read_csv("data/analytic/features.csv").with_columns(
        pl.col("eventname").replace(EVENT_MAPPING_1)
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
    df.select("Year", "p-factor", "Predictors", "Joined").drop_nulls().write_csv(
        "data/results/tables/follow_up.csv"
    )


def make_analysis_demographics():
    df = (
        pl.read_csv(
            "data/analytic/dataset.csv",
            columns=[
                "src_subject_id",
                "eventname",
                "p_factor",
                "demo_sex_v2_1",
                "interview_age",
            ],
        )
        .with_columns(pl.all().forward_fill().over("src_subject_id"))
        .with_columns(
            pl.col("eventname").replace(EVENT_MAPPING_2),
            pl.col("demo_sex_v2_1").replace(SEX_MAPPING),
            (pl.col("interview_age") / 12).round(0).cast(pl.Int32),
        )
    )
    race = pl.read_csv(
        "data/features/abcd_p_demo.csv",
        columns=["src_subject_id", "eventname", "race_ethnicity"],
    ).with_columns(
        pl.col("race_ethnicity").replace(RACE_MAPPING),
        pl.col("eventname").replace(EVENT_MAPPING_1),
    )
    df = df.join(race, on=["src_subject_id", "eventname"], how="inner")
    df.write_csv("data/demographics.csv")


def make_p_factor_group(df: pl.DataFrame, group_name: str, variable_name: str):
    return (
        df.group_by(group_name)
        .agg(
            (
                pl.col("p_factor").mean().round(3).cast(pl.Utf8)
                + " ± "
                + pl.col("p_factor").std().mul(2).round(2).cast(pl.Utf8)
            ).alias("p-factor"),
            pl.col("src_subject_id").n_unique().alias("Count"),
        )
        .sort(group_name)
        .rename({group_name: "Group"})
        .with_columns(
            pl.lit(variable_name).alias("Variable"), pl.col("Group").cast(pl.Utf8)
        )
    )


def p_factor_table():
    df = pl.read_csv("data/demographics.csv")
    dfs = [make_p_factor_group(df, column, name) for column, name in COLUMNS]
    total = (
        df.with_columns(
            pl.lit("All").alias("Group"),
            (
                pl.col("p_factor").mean().round(3).cast(pl.Utf8)
                + " ± "
                + pl.col("p_factor").std().mul(2).round(2).cast(pl.Utf8)
            ).alias("p-factor"),
            pl.col("src_subject_id").n_unique().alias("Count"),
            pl.lit("All").alias("Variable"),
        )
        .select("Group", "p-factor", "Count", "Variable")
        .head(1)
    )
    df = (
        pl.concat([total] + dfs)
        .select("Variable", "Group", "p-factor", "Count")
        .drop_nulls()
    )
    df.write_csv("data/results/tables/p_factors.csv")


# TODO
def metrics_table():
    pass


def apply_r2(args) -> pl.Series:
    return pl.Series([r2_score(y_true=args[0], y_pred=args[1])], dtype=pl.Float32)


def make_r2_group(df: pl.DataFrame, group_name: str, variable_name: str):
    return (
        df.group_by(group_name)
        .agg(
            pl.map_groups(exprs=["y_true", "y_pred"], function=apply_r2).alias("R2"),
            pl.col("src_subject_id").n_unique().alias("Subjects"),
        )
        .rename({group_name: "Group"})
        .sort("Group")
        .with_columns(
            pl.lit(variable_name).alias("Variable"), pl.col("Group").cast(pl.Utf8)
        )
        .select("Variable", "Group", "R2", "Subjects")
    )


def r2_table():
    df = pl.read_csv("data/results/predicted_vs_observed.csv")
    dfs = [make_r2_group(df, column, name) for column, name in COLUMNS]
    total = (
        df.with_columns(
            pl.lit("All").alias("Variable"),
            pl.lit("All").alias("Group"),
            pl.map_groups(exprs=["y_true", "y_pred"], function=apply_r2).alias("R2"),
            pl.col("src_subject_id").n_unique().alias("Subjects"),
        )
        .select("Variable", "Group", "R2", "Subjects")
        .head(1)
    )
    df = (
        pl.concat([total] + dfs)
        .with_columns(pl.col("R2").round(2))
        .filter(pl.col("R2") > 0)
    )
    df.write_csv("data/results/tables/r2_by_event.csv")


def make_tables():
    make_follow_up_table()
    p_factor_table()
    r2_table()


if __name__ == "__main__":
    make_analysis_demographics()
    make_follow_up_table()
    p_factor_table()
    r2_table()
