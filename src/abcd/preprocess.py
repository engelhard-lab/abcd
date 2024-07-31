from collections import defaultdict
from typing import Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import polars as pl
from polars.type_aliases import JoinStrategy
import polars.selectors as cs

from functools import partial, reduce

from abcd.config import Config


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
        .to_dummies("demo_sex_v2")
        .drop("demo_sex_v2_2", "demo_sex_v2_3")
    )
    return df


def make_adi(df: pl.DataFrame, join_on: list[str]):
    adi_columns = [
        "reshist_addr1_adi_perc",
        "reshist_addr2_adi_perc",
        "reshist_addr3_adi_perc",
    ]
    return (
        df.with_columns(
            pl.mean_horizontal(adi_columns).forward_fill().alias("adi_percentile")
        )
        .select(*join_on, "adi_percentile")
        .drop(adi_columns)
    )


def pad_dataframe(df: pl.DataFrame):
    return (
        df.group_by("src_subject_id", maintain_order=True)
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
            "led_l_adi": partial(make_adi, join_on=config.join_on),
            "abcd_y_lt": lambda df: df.to_dummies(["site_id_l"], drop_first=True),
        }
    )
    dfs = []
    for filename, metadata in config.features.model_dump().items():
        df = pl.read_csv(
            source=config.filepaths.data.raw.features / f"{filename}.csv",
            null_values=["", "null"],
            infer_schema_length=100_000,
            n_rows=2000 if config.fast_dev_run else None,
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
            .with_columns(pl.col("eventname").replace(EVENT_MAPPING).cast(pl.Int32))
            .filter(pl.col("eventname").ne(4))
            .pipe(transform)
            .pipe(drop_null_columns)
        )
        dfs.append(df)
    return dfs


def make_dataset(dfs: list[pl.DataFrame], config: Config):
    df = join_dataframes(dfs=dfs, join_on=config.join_on, how="outer_coalesce").sort(
        config.join_on
    )
    return df


def make_subject_metadata(config: Config, df: pl.DataFrame) -> pl.DataFrame:
    dfs = df.partition_by("src_subject_id", maintain_order=True)
    splits = split_data(dfs, config=config)
    splits = transform_splits(splits=splits, config=config, analysis="metadata")
    df = pl.concat(list(splits.values()))
    rename_mapping = {
        "src_subject_id": "Subject ID",
        "eventname": "Time",
        "p_factor_quartile": "Quartile at t",
        "label": "Quartile at t+1",
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
            pl.col("Time").replace(EVENT_MAPPING),
            pl.col("Sex").replace(SEX_MAPPING),
            pl.col("Race").replace(RACE_MAPPING),
            pl.col("Age").truediv(12).round(0).cast(pl.Int32),
            pl.col("ADI quartile").qcut(quantiles=4, labels=["1", "2", "3", "4"]),
        )
        .with_columns(
            pl.exclude("Subject ID", "Time", "Qartile at t", "Quartile at t+1", "Age")
            .forward_fill()
            .over("Subject ID")
        )
    )


class QuartileTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pl.DataFrame, y=None):
        self.quartiles: list = [
            X["factoranalysis0"].quantile(quantile=q) for q in [0.25, 0.5, 0.75]
        ]
        return self

    def transform(self, X: pl.DataFrame):
        return X.select(
            (
                pl.col("factoranalysis0")
                .cut(
                    breaks=self.quartiles,
                    labels=["1", "2", "3", "4"],
                )
                .cast(pl.Int32)
                .alias("p_factor_quartile")
            )
        )["p_factor_quartile"]


def split_data(dfs: list[pl.DataFrame], config: Config):
    X_train, X_val_test = train_test_split(
        dfs,
        train_size=config.preprocess.train_size,
        random_state=config.random_seed,
        shuffle=True,
    )
    X_val, X_test = train_test_split(
        X_val_test, train_size=0.5, random_state=config.random_seed, shuffle=False
    )
    return {
        "train": pl.concat(X_train),
        "val": pl.concat(X_val),
        "test": pl.concat(X_test),
    }


def post_process(X: pl.DataFrame, y, groups):
    return (
        X.with_columns(y, groups)
        .with_columns(
            pl.col("p_factor_quartile").shift(-1).over("src_subject_id").alias("label")
        )
        .filter(pl.col("eventname").ne(4))
        .pipe(pad_dataframe)
        .select(
            pl.col("src_subject_id", "eventname", "p_factor_quartile", "label"),
            pl.exclude("src_subject_id", "eventname", "p_factor_quartile", "label"),
        )
    )


def transform_features(
    splits: dict[str, pl.DataFrame], features, labels, feature_pipeline, label_pipeline
):
    for name, split in splits.items():
        groups = split.drop_in_place("src_subject_id")
        X = split.select(features)
        y = split.select(labels)
        if name == "train":
            X = feature_pipeline.fit_transform(X)
            y = label_pipeline.fit_transform(y)
        else:
            X = feature_pipeline.transform(X)
            y = label_pipeline.transform(y)
        splits[name] = post_process(X=X, y=y, groups=groups)
    return splits


def transform_by_year(
    splits: dict[str, pl.DataFrame], features, labels, feature_pipeline, label_pipeline
):
    for name, split in splits.items():
        groups = split.drop_in_place("src_subject_id")
        X = split.select(features)
        if name == "train":
            X = feature_pipeline.fit_transform(X)
        else:
            X = feature_pipeline.transform(X)
        pipelines = {}
        label_groups = {}
        for year, group in split.group_by("eventname", maintain_order=True):
            label_data = group.select(labels)
            if name == "train":
                label_pipeline = label_pipeline()
                label_groups[year] = label_pipeline.fit_transform(label_data)
                pipelines[year] = label_pipeline
            else:
                label_groups[year] = pipelines[year].transform(label_data)
        y = pl.concat(list(label_groups.values()))
        df = post_process(X=X, y=y, groups=groups)
        splits[name] = df
    return splits


def transform_metadata(splits, features, labels, label_pipeline):
    for name, split in splits.items():
        groups = split.drop_in_place("src_subject_id")
        X = split.select(features)
        y = split.select(labels)
        if name == "train":
            y = label_pipeline.fit_transform(y)
        else:
            y = label_pipeline.transform(y)
        splits[name] = post_process(X=X, y=y, groups=groups)
    return splits


def get_features_and_labels(config: Config, analysis: str):
    brain_datasets = (
        "mri_y_dti_fa_fs_at",
        "mri_y_rsfmr_cor_gp_gp",
        "mri_y_tfmr_sst_csvcg_dsk",
        "mri_y_tfmr_mid_alrvn_dsk",
        "mri_y_tfmr_nback_2b_dsk",
    )
    match analysis:
        case "with_brain" | "by_year" | "metadata":
            features = pl.exclude(config.features.mh_p_cbcl.columns)
        case "without_brain":
            brain_features = [
                column
                for name, features in config.features.model_dump().items()
                for column in features["columns"]
                if name in brain_datasets
            ]
            features = pl.exclude(config.features.mh_p_cbcl.columns + brain_features)
        case "symptoms" | "autoregressive":
            features = config.features.mh_p_cbcl.columns
        case "all":
            features = pl.all()
        case _:
            raise ValueError(f"Invalid analysis: {analysis}")
    return features, config.features.mh_p_cbcl.columns


def transform_splits(splits: dict, config: Config, analysis: str):
    feature_pipeline = make_pipeline(
        StandardScaler(), KNNImputer(n_neighbors=config.preprocess.n_neighbors)
    )
    label_pipeline = partial(
        make_pipeline,
        StandardScaler(),
        KNNImputer(n_neighbors=config.preprocess.n_neighbors),
        FactorAnalysis(n_components=1, random_state=config.random_seed),
        QuartileTransformer(),
    )
    features, labels = get_features_and_labels(config=config, analysis=analysis)
    if analysis == "by_year":
        splits = transform_by_year(
            splits=splits,
            features=features,
            labels=labels,
            feature_pipeline=feature_pipeline,
            label_pipeline=label_pipeline,
        )
    elif analysis == "metadata":
        splits = transform_metadata(
            splits=splits,
            features=features,
            labels=config.features.mh_p_cbcl.columns,
            label_pipeline=label_pipeline(),
        )
    else:
        splits = transform_features(
            splits=splits,
            features=features,
            labels=config.features.mh_p_cbcl.columns,
            feature_pipeline=feature_pipeline,
            label_pipeline=label_pipeline(),
        )
    return splits


def make_splits(
    dfs: list[pl.DataFrame], config: Config, analysis: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    splits = split_data(dfs, config=config)
    splits = transform_splits(splits=splits, config=config, analysis=analysis)
    train, val, test = splits.values()
    train.write_csv(config.filepaths.data.analytic.train)
    val.write_csv(config.filepaths.data.analytic.val)
    test.write_csv(config.filepaths.data.analytic.test)
    return train, val, test


def get_splits(
    df: pl.DataFrame, config: Config, analysis: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    filepaths = config.filepaths.data.analytic.model_dump().values()
    not_all_files_exist = not all([filepath.exists() for filepath in filepaths])
    if not_all_files_exist or config.regenerate:
        dfs = df.partition_by("src_subject_id", maintain_order=True)
        train, val, test = make_splits(dfs=dfs, config=config, analysis=analysis)
    else:
        train = pl.read_csv(config.filepaths.data.analytic.train)
        val = pl.read_csv(config.filepaths.data.analytic.val)
        test = pl.read_csv(config.filepaths.data.analytic.test)
    return train, val, test
