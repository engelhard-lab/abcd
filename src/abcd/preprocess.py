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
# import pandas as pd

from functools import reduce


from abcd.config import Config
from abcd.metadata import make_variable_metadata

EVENT_MAPPING = {
    "baseline_year_1_arm_1": 0,
    "1_year_follow_up_y_arm_1": 1,
    "2_year_follow_up_y_arm_1": 2,
    "3_year_follow_up_y_arm_1": 3,
    "4_year_follow_up_y_arm_1": 4,
}


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
        df.with_columns(
            pl.max_horizontal(education).alias("parent_highest_education"),
        )
        .with_columns(
            pl.all().forward_fill().backward_fill().over("src_subject_id"),
        )
        .drop(education)
        .to_dummies("demo_sex_v2", drop_first=True)
    )
    return df


def make_sites(df: pl.DataFrame):
    return df.to_dummies("site_id_l", drop_first=True)


def make_adi(df: pl.DataFrame):
    columns = [
        "reshist_addr1_adi_perc",
        "reshist_addr2_adi_perc",
        "reshist_addr3_adi_perc",
    ]
    return df.with_columns(pl.mean_horizontal(columns).alias("adi_percentile")).drop(
        columns
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
            "abcd_y_lt": make_sites,
            "led_l_adi": make_adi,
        }
    )
    dfs = []
    for filename, metadata in config.features.model_dump().items():
        df = pl.read_csv(
            source=config.filepaths.features / f"{filename}.csv",
            null_values=["", "null"],
            infer_schema_length=100_000,
        )
        if len(metadata["columns"]) > 0:
            columns = pl.col(config.join_on + metadata["columns"])
        else:
            columns = df.columns
        transform = transforms[filename]
        df = (
            df.select(columns)
            .filter(pl.col("eventname").is_in(EVENT_MAPPING.keys()))
            .select(~cs.contains(columns_to_drop))
            .with_columns(pl.all().replace({777: None, 999: None}))
            .pipe(transform)
            .pipe(drop_null_columns)
        )
        dfs.append(df)
    return dfs


def make_labels(config: Config) -> pl.DataFrame:
    df = pl.read_csv(
        "data/labels/mh_p_cbcl.csv", columns=config.labels.cbcl_labels + config.join_on
    ).sort(config.join_on)
    pipeline = make_pipeline(
        KNNImputer(n_neighbors=5),
        FactorAnalysis(
            n_components=1,
            random_state=config.random_seed,
        ),
    )
    values = df.select(config.labels.cbcl_labels).to_numpy()
    values = pipeline.fit_transform(values)
    p_factors = pl.Series(values)
    bin_labels = [str(i) for i in range(config.n_quantiles)]
    df = (
        df.with_columns(p_factor=p_factors)
        .with_columns(
            pl.col("p_factor").qcut(
                quantiles=config.n_quantiles,
                labels=bin_labels,
                allow_duplicates=True,
            )
        )
        .with_columns(
            pl.col("p_factor")
            .shift(-1)
            .over("src_subject_id")
            .cast(pl.Int32)
            .alias("next_p_factor")
        )
        .drop_nulls()
        .select(pl.exclude(config.labels.cbcl_labels))
    )
    return df


def make_dataset(dfs: list[pl.DataFrame], config: Config):
    features = join_dataframes(dfs=dfs, join_on=config.join_on, how="outer_coalesce")
    labels = make_labels(config=config)
    df = (
        labels.join(other=features, on=config.join_on, how="inner")
        .with_columns(pl.col("eventname").replace(EVENT_MAPPING))
        .sort(config.join_on)
    )
    df.write_csv("data/analytic/dataset.csv")
    return df.partition_by("src_subject_id", maintain_order=True)


def make_splits(df: list[pl.DataFrame], config: Config):
    X_train, X_val_test = train_test_split(
        df, train_size=config.train_size, random_state=config.random_seed, shuffle=True
    )
    X_val, X_test = train_test_split(
        X_val_test, train_size=0.5, random_state=config.random_seed, shuffle=False
    )
    X_train: pl.DataFrame = pl.concat(X_train)  # type: ignore
    X_val: pl.DataFrame = pl.concat(X_val)  # type: ignore
    X_test: pl.DataFrame = pl.concat(X_test)  # type: ignore
    X_test.write_csv("data/test_untransformed.csv")
    y_train = X_train.drop_in_place("next_p_factor")
    y_val = X_val.drop_in_place("next_p_factor")
    y_test = X_test.drop_in_place("next_p_factor")
    # quartile = X_test.select("src_subject_id", "p_factor")
    # quartile.write_csv("data/test_quartiles.csv")
    quartile_train = X_train.drop_in_place("p_factor")
    quartile_val = X_val.drop_in_place("p_factor")
    quartile_test = X_test.drop_in_place("p_factor")
    group_train = X_train.drop_in_place("src_subject_id")
    group_val = X_val.drop_in_place("src_subject_id")
    group_test = X_test.drop_in_place("src_subject_id")
    pipeline = make_pipeline(StandardScaler(), KNNImputer(n_neighbors=5))
    X_train = pipeline.fit_transform(X_train)  # type: ignore
    X_val = pipeline.transform(X_val)  # type: ignore
    X_test = pipeline.transform(X_test)  # type: ignore
    train = X_train.with_columns(
        src_subject_id=group_train, label=y_train, quartile=quartile_train
    )
    val = X_val.with_columns(
        src_subject_id=group_val, label=y_val, quartile=quartile_val
    )
    test = X_test.with_columns(
        src_subject_id=group_test, label=y_test, quartile=quartile_test
    )
    return train, val, test


def generate_data(config: Config):
    dfs = get_datasets(config=config)
    make_variable_metadata(dfs=dfs, features=config.features)
    df = make_dataset(dfs, config=config)
    train, val, test = make_splits(df, config=config)
    train.write_csv(config.filepaths.train)
    val.write_csv(config.filepaths.val)
    test.write_csv(config.filepaths.test)
    return train, val, test


def get_data(config: Config):
    paths = [config.filepaths.train, config.filepaths.val, config.filepaths.test]
    not_all_files_exist = not all([filepath.exists() for filepath in paths])
    if not_all_files_exist or config.regenerate:
        train, val, test = generate_data(config=config)
    else:
        train = pl.read_csv(config.filepaths.train)
        val = pl.read_csv(config.filepaths.val)
        test = pl.read_csv(config.filepaths.test)
    return train, val, test
