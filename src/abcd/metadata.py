import polars as pl

from abcd.config import Features


def rename_questions() -> pl.Expr:
    return (
        pl.when(pl.col("variable").str.contains("total_core"))
        .then(pl.lit("Adverse childhood experiences"))
        .when(pl.col("variable").str.contains("adi_percentile"))
        .then(pl.lit("Area deprivation index percentile"))
        .when(pl.col("variable").str.contains("parent_highest_education"))
        .then(pl.lit("Parent highest education"))
        .when(pl.col("variable").str.contains("demo_comb_income_v2"))
        .then(pl.lit("Household income"))
        .when(pl.col("variable").eq(pl.lit("eventname")))
        .then(pl.lit("Year"))
        .otherwise(pl.col("question"))
        .alias("question")
    )


def rename_datasets() -> pl.Expr:
    return (
        pl.when(pl.col("variable").str.contains("eventname|site_id"))
        .then(pl.lit("Site and year"))
        .when(pl.col("variable").str.contains("demo_sex_v2_|interview_age"))
        .then(pl.lit("Age and sex"))
        .when(
            pl.col("variable").str.contains(
                "adi_percentile|demo_comb_income_v2|parent_highest_education"
            )
        )
        .then(pl.lit("Socio-economic status & area deprivation"))
        .otherwise(pl.col("dataset"))
        .alias("dataset")
    )


def make_variable_df(dfs: list[pl.DataFrame], features: Features) -> pl.DataFrame:
    metadata_dfs: list[pl.DataFrame] = []
    for df, (filename, metadata) in zip(dfs, features.model_dump().items()):
        table_metadata = {"table": [], "dataset": [], "respondent": [], "variable": []}
        for column in df.columns:
            table_metadata["table"].append(filename)
            table_metadata["dataset"].append(metadata["name"])
            table_metadata["respondent"].append(metadata["respondent"])
            table_metadata["variable"].append(column)
            metadata_df = pl.DataFrame(table_metadata)
        metadata_dfs.append(metadata_df)
    return pl.concat(metadata_dfs)


def captialize(column: str) -> pl.Expr:
    return pl.col(column).str.slice(0, 1).str.to_uppercase() + pl.col(column).str.slice(
        1
    )


def format_questions() -> pl.Expr:
    return (
        pl.col("question")
        .str.replace("\\..*|(!s)/(!g).*|\\?.*", "")
        .str.to_lowercase()
        .str.slice(0)
    )


def make_variable_metadata(dfs: list[pl.DataFrame], features: Features):
    variables = make_variable_df(dfs=dfs, features=features)
    questions = (
        pl.read_csv(
            "data/abcd_data_dictionary.csv",
            columns=["table_name", "var_name", "var_label", "notes"],
        )
        .rename(
            {
                "table_name": "table",
                "var_name": "variable",
                "var_label": "question",
                "notes": "response",
            }
        )
        .drop("table_name")
    )
    df = (
        (
            variables.join(questions, on=["table", "variable"], how="left")
            .with_columns(
                format_questions(),
                pl.col("dataset").str.replace_all("_", " "),
                pl.col("response").str.replace_all("\\s*/\\s*[^;]+", ""),
            )
            .with_columns(
                captialize("dataset"),
                captialize("question"),
            )
            .with_columns(
                rename_questions(),
                rename_datasets(),
            )
        )
        .unique(subset=["variable"])
        .sort("dataset", "respondent", "variable")
    )
    df.write_csv("data/variables.csv")


def make_subject_metadata(splits: dict[str, pl.DataFrame]) -> pl.DataFrame:
    # RACE_MAPPING = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
    # SEX_MAPPING = {1: "Male", 2: "Female"}
    df = pl.concat(
        [
            split.with_columns(pl.lit(name).alias("Split"))
            for name, split in splits.items()
        ]
    )
    race = pl.read_csv(
        "data/raw/features/abcd_p_demo.csv",
        columns=["src_subject_id", "race_ethnicity"],
    )
    df = df.join(race, on="src_subject_id", how="left")
    sex_mapping = dict(
        zip(
            df["demo_sex_v2_1"]
            .unique(maintain_order=True)
            .sort(descending=True)
            .to_list(),
            ["Male", "Female"],
        )
    )
    race_mapping = dict(
        zip(
            df["race_ethnicity"]
            .unique(maintain_order=True)
            .drop_nulls()
            .sort(descending=True)
            .to_list(),
            ["White", "Black", "Hispanic", "Asian", "Other"],
        )
    )
    print(sex_mapping, race_mapping)
    rename_mapping = {
        "src_subject_id": "Subject ID",
        "eventname": "Time",
        # "p_factor_by_year": "Quartile at t by year",
        # "label_by_year": "Quartile at t+1 by year",
        "y_t": "Quartile at t",
        "y_{t+1}": "Quartile at t+1",
        "demo_sex_v2_1": "Sex",
        "race_ethnicity": "Race",
        "interview_age": "Age",
        "adi_percentile": "ADI quartile",
        "parent_highest_education": "Parent highest education",
        "demo_comb_income_v2": "Combined income",
    }
    return (
        df.rename(rename_mapping)
        .select("Split", *rename_mapping.values())
        .with_columns(
            pl.col("Sex").replace(sex_mapping),
            pl.col("Race").replace(race_mapping),
            pl.col("Age").truediv(12).round(0).cast(pl.Int32),
            pl.col("ADI quartile").qcut(quantiles=4, labels=["1", "2", "3", "4"]),
        )
        .with_columns(
            pl.exclude(
                "Subject ID",
                "Time",
                "Qartile at t",
                "Quartile at t+1",
                "Quartile at t by year",
                "Quartile at t+1 by year",
                "Age",
            )
            .forward_fill()
            .over("Subject ID")
        )
    )
