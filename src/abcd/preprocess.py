from collections import defaultdict
from typing import Callable
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import polars as pl
from polars.type_aliases import JoinStrategy
import polars.selectors as cs
import pandas as pd

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
            .select(~cs.contains(columns_to_drop))
            .with_columns(pl.all().replace({777: None, 999: None}))
            .pipe(transform)
            .pipe(drop_null_columns)
        )
        dfs.append(df)
    return dfs


def make_labels(config: Config) -> pl.DataFrame:
    targets = ["abcl_scr_prob_internal_r", "abcl_scr_prob_external_r"]
    df = pl.read_csv(
        "data/labels/mh_p_cbcl.csv", columns=targets + config.join_on
    ).sort(config.join_on)  # [config.target]
    pipeline = make_pipeline(
        StandardScaler(),
        FactorAnalysis(n_components=1, random_state=config.random_seed),
    )
    p_factors = pl.Series(pipeline.fit_transform(df.select(targets).to_numpy()))
    # bin_labels = [str(i) for i in range(config.n_quantiles)]
    return (
        df.with_columns(p_factor=p_factors)
        .with_columns(
            pl.col(config.target)
            # .qcut(
            #     quantiles=config.n_quantiles,
            #     labels=bin_labels,
            #     allow_duplicates=True,
            # )
            .shift(-1)
            .over("src_subject_id")
            # .cast(pl.Int32)
        )
        .drop_nulls()
    )


def make_dataset(dfs: list[pl.DataFrame], config: Config):
    features = join_dataframes(dfs=dfs, join_on=config.join_on, how="outer_coalesce")
    features.write_csv("data/analytic/features.csv")
    labels = make_labels(config=config)
    df = (
        labels.join(other=features, on=config.join_on, how="inner")
        .with_columns(pl.col("eventname").replace(EVENT_MAPPING))
        .sort(config.join_on)
    )
    df.write_csv("data/analytic/dataset.csv")
    return df.partition_by("src_subject_id", include_key=True)


def make_splits(df, config: Config):
    X_train, X_val_test = train_test_split(
        df, train_size=config.train_size, random_state=config.random_seed, shuffle=True
    )
    X_val, X_test = train_test_split(
        X_val_test, train_size=0.5, random_state=config.random_seed, shuffle=True
    )
    X_train = pl.concat(X_train).to_pandas()
    X_val = pl.concat(X_val).to_pandas()
    X_test = pl.concat(X_test).to_pandas()
    X_test.to_csv("data/test_untransformed.csv", index=False)
    y_train = X_train.pop("p_factor")
    y_val = X_val.pop("p_factor")
    y_test = X_test.pop("p_factor")
    group_train = X_train.pop("src_subject_id")
    group_val = X_val.pop("src_subject_id")
    group_test = X_test.pop("src_subject_id")
    pipeline_steps = [
        ("scaler", StandardScaler()),
        ("imputer", KNNImputer(n_neighbors=5)),
    ]
    pipeline = Pipeline(steps=pipeline_steps)
    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)
    X_test = pipeline.transform(X_test)
    train = pd.concat([group_train, y_train, X_train], axis=1)  # type: ignore
    val = pd.concat([group_val, y_val, X_val], axis=1)  # type: ignore
    test = pd.concat([group_test, y_test, X_test], axis=1)  # type: ignore
    return train, val, test


def generate_data(config: Config):
    dfs = get_datasets(config=config)
    make_variable_metadata(dfs=dfs, features=config.features)
    df = make_dataset(dfs, config=config)
    train, val, test = make_splits(df, config=config)
    train.to_csv(config.filepaths.train, index=False)
    val.to_csv(config.filepaths.val, index=False)
    test.to_csv(config.filepaths.test, index=False)
    return train, val, test


def get_data(config: Config):
    paths = [config.filepaths.train, config.filepaths.val, config.filepaths.test]
    not_all_files_exist = not all([filepath.exists() for filepath in paths])
    if not_all_files_exist or config.regenerate:
        return generate_data(config=config)
    else:
        train = pd.read_csv(config.filepaths.train)
        val = pd.read_csv(config.filepaths.val)
        test = pd.read_csv(config.filepaths.test)
        return train, val, test
