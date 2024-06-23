from collections import defaultdict
from typing import Callable
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import polars as pl
from polars.type_aliases import JoinStrategy
import polars.selectors as cs

from functools import reduce


from abcd.config import Config
from abcd.metadata import make_variable_metadata


RACE_MAPPING = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
SEX_MAPPING = {1: "Male", 0: "Female"}
EVENTS = [
    "baseline_year_1_arm_1",
    "1_year_follow_up_y_arm_1",
    "2_year_follow_up_y_arm_1",
    "3_year_follow_up_y_arm_1",
    "4_year_follow_up_y_arm_1",
]
EVENT_INDEX = list(range(len(EVENTS)))
EVENT_MAPPING = dict(zip(EVENTS, EVENT_INDEX))
EVENT_NAMES = ["Baseline", "Year 1", "Year 2", "Year 3", "Year 4"]
REVERSE_EVENT_MAPPING = dict(zip(EVENT_INDEX, EVENT_NAMES))


def drop_null_columns(features: pl.DataFrame, cutoff=0.25) -> pl.DataFrame:
    null_proportion = features.null_count() / features.shape[0]
    columns_to_keep = (null_proportion < cutoff).transpose().to_series()
    return features.select(
        [col for col, keep in zip(features.columns, columns_to_keep) if keep]
    )


def join_dataframes(
    dfs: list[pl.DataFrame], join_on: list[str], how: JoinStrategy
) -> pl.DataFrame:
    return reduce(
        lambda left, right: left.join(
            right,
            how=how,
            on=join_on,
        ),
        dfs,
    )


def make_demographics(df: pl.DataFrame):
    education = ["demo_prnt_ed_v2", "demo_prtnr_ed_v2"]
    df = (
        df.with_columns(pl.max_horizontal(education).alias("parent_highest_education"))
        .drop(education)
        .with_columns(
            pl.all().forward_fill().backward_fill().over("src_subject_id"),
        )
        .to_dummies("demo_sex_v2", drop_first=True)
        .drop("demo_sex_v2_3")
    )
    return df


def make_adi(df: pl.DataFrame):
    columns = [
        "reshist_addr1_adi_perc",
        "reshist_addr2_adi_perc",
        "reshist_addr3_adi_perc",
    ]
    return df.with_columns(
        pl.mean_horizontal(columns).forward_fill().alias("adi_percentile")
    ).drop(columns)


def pad_dataframe(df: pl.DataFrame):
    return (
        df.group_by("src_subject_id")
        .agg(pl.all().extend_constant(None, len(EVENTS) - pl.len()))
        .with_columns(pl.lit(EVENTS).alias("eventname"))
        .explode(pl.exclude("src_subject_id"))
    )


def get_datasets(config: Config) -> list[pl.DataFrame]:
    columns_to_drop = (
        "_nm",
        "_nt",
        "_na",
        "_language",
        "_answered",
        "ss_sbd",
        "ss_da",
        "_total",
        "_mean",
        "sds_",
        "srpf_",
        "_fc",
    )
    transforms: defaultdict[str, Callable[[pl.DataFrame], pl.DataFrame]] = defaultdict(
        lambda: lambda df: df
    )
    transforms.update(
        {
            "abcd_p_demo": make_demographics,
            "led_l_adi": make_adi,
            "abcd_y_lt": lambda df: df.to_dummies(["site_id_l"], drop_first=True),
        }
    )
    dfs = []
    for filename, metadata in config.features.model_dump().items():
        df = pl.read_csv(
            source=config.filepaths.features / f"{filename}.csv",
            null_values=["", "null"],
            infer_schema_length=100_000,
        )
        if metadata["columns"]:
            columns = pl.col(config.join_on + metadata["columns"])
        else:
            columns = df.columns
        transform = transforms[filename]
        df = (
            df.select(columns)
            .select(~cs.contains(columns_to_drop))
            .filter(pl.col("eventname").is_in(EVENTS))
            .with_columns(pl.all().replace({777: None, 999: None}))
            .pipe(transform)
            .pipe(drop_null_columns)
            .pipe(pad_dataframe)
        )
        dfs.append(df)
    return dfs


def make_labels(config: Config) -> pl.DataFrame:
    df = pl.read_csv(
        "data/labels/mh_p_cbcl.csv", columns=config.labels.cbcl_labels + config.join_on
    ).sort(config.join_on)
    pipeline = make_pipeline(
        KNNImputer(n_neighbors=config.preprocess.n_neighbors),
        FactorAnalysis(n_components=1, random_state=config.random_seed),
    )
    values = df.select(config.labels.cbcl_labels).to_numpy()
    values = pipeline.fit_transform(values)
    p_factors = pl.Series(values)
    bin_labels = [str(i) for i in range(config.preprocess.n_quantiles)]
    df = (
        df.with_columns(
            p_factors.qcut(
                quantiles=config.preprocess.n_quantiles,
                labels=bin_labels,
                allow_duplicates=True,
            )
            .cast(pl.Int32)
            .alias("p_factor")
        )
        .with_columns(
            pl.col("p_factor")
            .shift(-1)
            .over("src_subject_id")
            .cast(pl.Int32)
            .alias("next_p_factor")
        )
        .select(pl.exclude(config.labels.cbcl_labels))
        .pipe(pad_dataframe)
    )
    return df


def make_dataset(dfs: list[pl.DataFrame], config: Config):
    features = join_dataframes(dfs=dfs, join_on=config.join_on, how="outer_coalesce")
    labels = make_labels(config=config)
    df = (
        labels.join(other=features, on=config.join_on, how="inner")
        .with_columns(pl.col("eventname").replace(EVENT_MAPPING).cast(pl.Int32))
        .sort(config.join_on)
        .filter(pl.col("eventname").ne(4))
    )
    df.write_csv("data/analytic/dataset.csv")
    return df.partition_by("src_subject_id", maintain_order=True)


def make_metadata(df: pl.DataFrame):
    rename_mapping = {
        "src_subject_id": "Subject ID",
        "eventname": "Measurement year",
        "p_factor": "Quartile",
        "next_p_factor": "Next quartile",
        "demo_sex_v2_1": "Sex",
        "race_ethnicity": "Race",
        "interview_age": "Age",
        "adi_percentile": "ADI quartile",
        "parent_highest_education": "Parent highest education",
        "demo_comb_income_v2": "Combined income",
    }
    return (
        df.rename(rename_mapping)
        .select(rename_mapping.values())
        .with_columns(
            pl.col("Measurement year").replace(REVERSE_EVENT_MAPPING),
            pl.col("Sex").replace(SEX_MAPPING),
            pl.col("Race").replace(RACE_MAPPING),
            pl.col("Age").truediv(12).round(0).cast(pl.Int32),
            pl.col("ADI quartile").qcut(quantiles=4, labels=["1", "2", "3", "4"]),
        )
    )


def make_splits(dfs: list[pl.DataFrame], config: Config):
    X_train, X_val_test = train_test_split(
        dfs,
        train_size=config.preprocess.train_size,
        random_state=config.random_seed,
        shuffle=True,
    )
    X_val, X_test = train_test_split(
        X_val_test, train_size=0.5, random_state=config.random_seed, shuffle=False
    )
    pipeline = make_pipeline(
        StandardScaler(), KNNImputer(n_neighbors=config.preprocess.n_neighbors)
    )
    splits = {"train": X_train, "val": X_val, "test": X_test}
    for name, split in splits.items():
        features: pl.DataFrame = pl.concat(split)  # type: ignore
        if name == "test":
            test_metadata = make_metadata(features)
            test_metadata.write_csv("data/test_metadata.csv")
        labels = features.drop_in_place("next_p_factor")
        features.drop_in_place("p_factor")
        features.drop_in_place("race_ethnicity")
        group = features.drop_in_place("src_subject_id")
        if name == "train":
            features = pipeline.fit_transform(features)  # type: ignore
        else:
            features = pipeline.transform(features)  # type: ignore
        splits[name] = features.with_columns(src_subject_id=group, label=labels)
    return splits


def generate_data(config: Config):
    dfs = get_datasets(config=config)
    make_variable_metadata(dfs=dfs, features=config.features)
    dfs = make_dataset(dfs, config=config)
    metadata = make_metadata(pl.concat(dfs))
    metadata.write_csv("data/metadata.csv")
    train, val, test = make_splits(dfs, config=config).values()
    train.write_csv(config.filepaths.train)
    val.write_csv(config.filepaths.val)
    test.write_csv(config.filepaths.test)
    return train, val, test


def get_data(config: Config):
    filepaths = [
        config.filepaths.train,
        config.filepaths.val,
        config.filepaths.test,
    ]
    not_all_files_exist = not all([filepath.exists() for filepath in filepaths])
    if not_all_files_exist or config.regenerate:
        train, val, test = generate_data(config=config)
    else:
        train, val, test = (pl.read_csv(filepath) for filepath in filepaths)
    return train, val, test
