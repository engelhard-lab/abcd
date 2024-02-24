from multiprocessing import cpu_count
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.decomposition import PCA
import polars as pl
import pandas as pd
import numpy as np
from polars.type_aliases import JoinStrategy
import polars.selectors as cs
import statsmodels.api as sm

from tomllib import load
from pathlib import Path
from functools import reduce

from abcdclinical.config import Config, Filepaths

EVENT_MAPPING = {
    "baseline_year_1_arm_1": 0,
    "1_year_follow_up_y_arm_1": 1,
    "2_year_follow_up_y_arm_1": 2,
    "3_year_follow_up_y_arm_1": 3,
    "4_year_follow_up_y_arm_1": 4,
}


def make_labels(filepath: Path) -> pl.DataFrame:
    return (
        pl.read_csv(filepath, null_values="NA")
        .with_columns(pl.col("p_score").shift(-1).over("src_subject_id"))
        .drop_nulls()
    )


def drop_null_columns(features, cutoff=0.25):
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


def process_demographics(columns, X_train, X_val, X_test):
    one_hot_encoder = OneHotEncoder(drop="if_binary", sparse_output=False)
    X_train["sex"] = one_hot_encoder.fit_transform(X_train.loc[:, ["sex"]]).to_numpy()  # type: ignore
    X_val["sex"] = one_hot_encoder.transform(X_val.loc[:, ["sex"]]).to_numpy()  # type: ignore
    X_test["sex"] = one_hot_encoder.transform(X_test.loc[:, ["sex"]]).to_numpy()  # type: ignore
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    X_train[columns] = pipeline.fit_transform(X_train[columns])
    X_val[columns] = pipeline.transform(X_val[columns])
    X_test[columns] = pipeline.transform(X_test[columns])
    return X_train, X_val, X_test


def get_datasets(filepath: Path):
    return [
        pl.read_csv(
            source=filepath, null_values=["", "null"], infer_schema_length=50_000
        )
        .select(
            ~cs.contains(
                (
                    "_nm",
                    "_nt",
                    "_na",
                    "_language",
                    "_answered",
                    "ss_sbd",
                    "ss_da",
                    "_total",
                    "_mean",
                )
            )
        )
        .pipe(drop_null_columns)
        for filepath in filepath.glob("*")
    ]


def make_dataset(dfs: list[pl.DataFrame], labels_path: Path, join_on: list[str]):
    labels = make_labels(filepath=labels_path)
    features = join_dataframes(dfs=dfs, join_on=join_on, how="outer_coalesce")
    features.write_csv("data/analytic/features.csv")
    df = labels.join(other=features, on=join_on, how="inner")
    df = df.with_columns(pl.col("eventname").replace(EVENT_MAPPING))
    df = df.drop("race_ethnicity")
    df.write_csv("data/analytic/dataset.csv")
    return df.partition_by("src_subject_id", include_key=True)


def process_dataset(df, n_components: int | None):
    X_train, X_val_test = train_test_split(
        df, train_size=0.8, random_state=1, shuffle=True
    )
    X_val, X_test = train_test_split(
        X_val_test, train_size=0.5, random_state=1, shuffle=True
    )
    X_train = pl.concat(X_train).to_pandas()
    X_val = pl.concat(X_val).to_pandas()
    X_test = pl.concat(X_test).to_pandas()
    demographic_columns = ["eventname", "sex", "age"]
    X_train, X_val, X_test = process_demographics(
        columns=demographic_columns, X_train=X_train, X_val=X_val, X_test=X_test
    )
    train_demographics = X_train[demographic_columns]
    val_demographics = X_val[demographic_columns]
    test_demographics = X_test[demographic_columns]
    X_train = X_train[X_train.columns.difference(demographic_columns)]
    X_val = X_val[X_val.columns.difference(demographic_columns)]
    X_test = X_test[X_test.columns.difference(demographic_columns)]
    y_train = X_train.pop("p_score")
    y_val = X_val.pop("p_score")
    y_test = X_test.pop("p_score")
    group_train = X_train.pop("src_subject_id")
    group_val = X_val.pop("src_subject_id")
    group_test = X_test.pop("src_subject_id")
    feature_names = X_train.columns
    pipeline_steps = [
        ("scaler", StandardScaler()),
        ("imputer", SimpleImputer(strategy="median")),
        ("pca", PCA(n_components=n_components)),
    ]
    pipeline = Pipeline(steps=pipeline_steps)
    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)
    X_test = pipeline.transform(X_test)
    X_train = sm.add_constant(X_train).rename({"const": "intercept"}, axis=1)  # type: ignore
    X_val = sm.add_constant(X_val).rename({"const": "intercept"}, axis=1)  # type: ignore
    X_test = sm.add_constant(X_test).rename({"const": "intercept"}, axis=1)  # type: ignore
    X_train = pd.concat([X_train, train_demographics], axis=1)  # type: ignore
    X_val = pd.concat([X_val, val_demographics], axis=1)  # type: ignore
    X_test = pd.concat([X_test, test_demographics], axis=1)  # type: ignore
    return (
        pipeline,
        feature_names,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        group_train,
        group_val,
        group_test,
    )


def make_column_mapping(join_on):
    dataset_mapping = {
        "ce_p_fes": "family_environment_parent",
        "ce_y_fes": "family_environment_youth",
        "ce_p_nsc": "neighborhood_parent",
        "ce_y_nsc": "neighborhood_youth",
        "ce_y_pm": "problem_monitor_youth",
        "ce_p_psb": "prosocial_parent",
        "ce_y_psb": "prosocial_youth",
        "ce_y_srpf": "school_youth",
        "nt_p_stq": "screentime_parent",
        "nt_y_st": "screentime_youth",
        "ph_p_sds": "sleep_disturbance_parent",
        "su_p_pr": "rules_parent",
        "dti_fa": "brain_dti_fa",
        "rsfmri": "brain_rsfmri",
        "sst": "brain_sst",
    }
    columns = {
        dataset_mapping[dataset]: pl.read_csv(
            "data/features/" + dataset + ".csv",
            null_values="",
            infer_schema_length=50_000,
        )
        .drop(join_on)
        .columns
        for dataset in dataset_mapping
    }
    return {value: key for key, values in columns.items() for value in values}


def get_components(pipeline, feature_names, join_on):
    components = pipeline.named_steps["pca"].components_.T
    component_names = range(1, components.shape[1] + 1)
    explained_variance = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pipeline.named_steps[
                "pca"
            ].explained_variance_ratio_,
        }
    )
    components = pd.DataFrame(components, columns=component_names)
    components["name"] = feature_names
    column_mapping = make_column_mapping(join_on)
    components["dataset"] = components["name"].map(column_mapping)
    return components, explained_variance


def generate_data(config, n_components: int | None):
    dfs = get_datasets(config.filepaths.features)
    join_on = ["src_subject_id", "eventname"]
    df = make_dataset(dfs, labels_path=config.filepaths.labels, join_on=join_on)
    (
        pipeline,
        feature_names,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        group_train,
        group_val,
        group_test,
    ) = process_dataset(df, n_components=n_components)
    components, explained_variance = get_components(
        pipeline=pipeline, feature_names=feature_names, join_on=join_on
    )
    explained_variance.to_csv("data/results/explained_variance.csv", index=False)  # type: ignore
    components.to_csv("data/results/principal_components.csv", index=False)  # type: ignore
    train = pd.concat([group_train, y_train, X_train], axis=1)
    val = pd.concat([group_val, y_val, X_val], axis=1)
    test = pd.concat([group_test, y_test, X_test], axis=1)
    train.to_csv("data/analytic/train.csv", index=False)
    val.to_csv("data/analytic/val.csv", index=False)
    test.to_csv("data/analytic/test.csv", index=False)
    return (
        X_train,
        y_train,
        group_train,
        X_val,
        y_val,
        group_val,
        X_test,
        y_test,
        group_test,
    )


def get_data(config: Config, regenerate: bool, n_components: int | None = None):
    paths = [config.filepaths.train, config.filepaths.val, config.filepaths.test]
    not_all_files_exist = not all([filepath.exists() for filepath in paths])
    if not_all_files_exist or regenerate:
        return generate_data(config=config, n_components=n_components)
    else:
        X_train = pd.read_csv(config.filepaths.train)
        X_val = pd.read_csv(config.filepaths.val)
        X_test = pd.read_csv(config.filepaths.test)
        y_train = X_train.pop("p_score")
        y_val = X_val.pop("p_score")
        y_test = X_test.pop("p_score")
        group_train = X_train.pop("src_subject_id")
        group_val = X_val.pop("src_subject_id")
        group_test = X_test.pop("src_subject_id")
        return (
            X_train,
            y_train,
            group_train,
            X_val,
            y_val,
            group_val,
            X_test,
            y_test,
            group_test,
        )


def drop_demographics(X_train, X_val, X_test):
    train_sex = X_train.pop("sex")
    val_sex = X_val.pop("sex")
    test_sex = X_test.pop("sex")
    train_age = X_train.pop("age")
    val_age = X_val.pop("age")
    test_age = X_test.pop("age")
    return X_train, X_val, X_test


if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = Config(**load(f))
    set_config(transform_output="pandas")
    (
        X_train,
        y_train,
        group_train,
        X_val,
        y_val,
        group_val,
        X_test,
        y_test,
        group_test,
    ) = get_data(config, regenerate=True)
