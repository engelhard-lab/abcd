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
    ).sort(join_on)


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
        .agg(pl.all().extend_constant(None, len(EVENT_INDEX) - pl.len()))
        .with_columns(pl.lit(EVENT_INDEX).alias("eventname"))
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
            .with_columns(
                pl.col("eventname")
                .replace(EVENT_MAPPING)
                .cast(pl.Int32)
                .alias("eventname")
            )
            .pipe(transform)
            .pipe(drop_null_columns)
        )
        dfs.append(df)
    return dfs


def make_subject_metadata(splits: dict[str, pl.DataFrame]) -> pl.DataFrame:
    dfs = [
        split.clone().with_columns(pl.lit(name).alias("Split"))
        for name, split in splits.items()
    ]
    df: pl.DataFrame = pl.concat(dfs)
    rename_mapping = {
        "src_subject_id": "Subject ID",
        "eventname": "Time",
        "p_factor_by_year": "Quartile at t by year",
        "label_by_year": "Quartile at t+1 by year",
        "p_factor": "Quartile at t",
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
        .select(["Split"] + list(rename_mapping.values()))
        .with_columns(
            pl.col("Sex").replace(SEX_MAPPING),
            pl.col("Race").replace(RACE_MAPPING),
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


class QuartileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        self.cutpoints = [0.25, 0.5, 0.75]
        self.labels = ["0", "1", "2", "3"]
        self.quartiles: pl.Series

    def fit(self, X: pl.DataFrame, y=None):
        self.quartiles = pl.Series(
            [X["factoranalysis0"].quantile(quantile=q) for q in self.cutpoints]
        )
        self.quartiles = self.quartiles.unique(maintain_order=True)
        self.labels = self.labels[: self.quartiles.len() + 1]
        return self

    def transform(self, X: pl.DataFrame):
        return X.select(
            (
                pl.col("factoranalysis0")
                .cut(
                    breaks=self.quartiles.to_list(),
                    labels=self.labels,
                )
                .cast(pl.Int32)
                .alias(self.column)
            )
        )


def split_data(df: pl.DataFrame, config: Config):
    dfs = df.partition_by("src_subject_id", maintain_order=True)
    train, val_test = train_test_split(
        dfs,
        train_size=config.preprocess.train_size,
        random_state=config.random_seed,
        shuffle=True,
    )
    val, test = train_test_split(
        val_test, train_size=0.5, random_state=config.random_seed, shuffle=False
    )
    return {
        "train": pl.concat(train),
        "val": pl.concat(val),
        "test": pl.concat(test),
    }


def transform_split(name: str, split: pl.DataFrame, columns, pipeline):
    data = split.select(columns)
    if name == "train":
        transformed_data = pipeline.fit_transform(data)
    else:
        transformed_data = pipeline.transform(data)
    transformed_split = split.with_columns(transformed_data)
    return transformed_split


def transform_by_group(
    name: str, split: pl.DataFrame, columns, groups, pipeline, pipelines
):
    dfs = []
    for group_name, group in split.group_by(
        groups, maintain_order=True, include_key=False
    ):
        data = group.select(columns)
        if name == "train":
            group_pipeline = pipeline()
            transformed_data = group_pipeline.fit_transform(data)
            pipelines[group_name] = group_pipeline
        else:
            group_pipeline = pipelines[group_name]
            transformed_data = group_pipeline.transform(data)
        df = group.with_columns(transformed_data)
        dfs.append(df)
    df = pl.concat(dfs)
    split = split.with_columns(df)
    return split


def make_label_pipeline(name, config):
    return partial(
        make_pipeline,
        FactorAnalysis(n_components=1, random_state=config.random_seed),
        QuartileTransformer(name),
    )


def shift_quartile(p_factor, label):
    return pl.col(p_factor).shift(-1).over("src_subject_id").alias(label)


def impute_nulls(df: pl.DataFrame):
    # 1. Forward fill within each subject
    # 2. Back fill if some subjects start with nulls
    # 3. Fill nulls with mean across all subjects if some subjects have all nulls
    return df.with_columns(
        pl.all().forward_fill().backward_fill().over("src_subject_id")
    ).with_columns(cs.numeric().fill_null(cs.numeric().median()))


def add_labels(splits: dict[str, pl.DataFrame], config: Config):
    label_pipeline = make_label_pipeline("p_factor", config)
    pipelines = {}
    for name, split in splits.items():
        split = split.filter(
            ~pl.all_horizontal(pl.col(config.features.mh_p_cbcl.columns).is_null())
        )
        split = impute_nulls(df=split)
        split = transform_by_group(
            name=name,
            split=split,
            columns=config.features.mh_p_cbcl.columns,
            groups="eventname",
            pipeline=make_label_pipeline("p_factor_by_year", config),
            pipelines=pipelines,
        ).with_columns(
            shift_quartile(p_factor="p_factor_by_year", label="label_by_year")
        )
        split = (
            transform_split(
                name=name,
                split=split,
                columns=config.features.mh_p_cbcl.columns,
                pipeline=label_pipeline(),
            )
            .with_columns(shift_quartile(p_factor="p_factor", label="label"))
            .pipe(pad_dataframe)
            .filter(
                pl.col("eventname").ne(4),
                ~pl.col("label").is_null().all().over("src_subject_id"),
            )
        )
        split.write_csv(getattr(config.filepaths.data.raw.splits, name))
        splits[name] = split
    return splits


def add_features(features, splits: dict[str, pl.DataFrame], analysis: str):
    pipeline = StandardScaler()
    label = "label_by_year" if analysis == "by_year" else "label"
    for name, split in splits.items():
        split = transform_split(
            name=name, split=split, columns=features, pipeline=pipeline
        ).select("src_subject_id", pl.col(label).alias("label"), features)
        splits[name] = split
    return splits


def get_brain_features(config: Config):
    brain_datasets = (
        "mri_y_dti_fa_fs_at",
        "mri_y_rsfmr_cor_gp_gp",
        "mri_y_tfmr_sst_csvcg_dsk",
        "mri_y_tfmr_mid_alrvn_dsk",
        "mri_y_tfmr_nback_2b_dsk",
    )
    return [
        column
        for name, features in config.features.model_dump().items()
        for column in features["columns"]
        if name in brain_datasets
    ]


def get_features(analysis: str, config: Config):
    match analysis:
        case "with_brain" | "by_year":
            features = cs.exclude(config.features.mh_p_cbcl.columns)
        case "without_brain":
            brain_features = get_brain_features(config)
            features = cs.exclude(config.features.mh_p_cbcl.columns + brain_features)
        case "symptoms":
            features = cs.by_name(config.features.mh_p_cbcl.columns)
        case "all":
            features = cs.all()
        case _:
            raise ValueError(f"Invalid analysis: {analysis}")
    return features & cs.exclude(
        "src_subject_id",
        "race_ethnicity",
        "p_factor",
        "label",
        "p_factor_by_year",
        "label_by_year",
    )


def process_splits(
    splits: dict, config: Config, analysis: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    filepaths = config.filepaths.data.analytic.model_dump().values()
    not_all_files_exist = not all([filepath.exists() for filepath in filepaths])
    if not_all_files_exist or config.regenerate:
        splits = {name: split.clone() for name, split in splits.items()}
        features = get_features(analysis=analysis, config=config)
        splits = add_features(splits=splits, features=features, analysis=analysis)
        train, val, test = splits.values()
        train.write_csv(config.filepaths.data.analytic.train)
        val.write_csv(config.filepaths.data.analytic.val)
        test.write_csv(config.filepaths.data.analytic.test)
    else:
        train = pl.read_csv(config.filepaths.data.analytic.train)
        val = pl.read_csv(config.filepaths.data.analytic.val)
        test = pl.read_csv(config.filepaths.data.analytic.test)
    return train, val, test


def get_raw_dataset(config: Config):
    if config.regenerate:
        datasets = get_datasets(config=config)
        make_variable_metadata(dfs=datasets, features=config.features)
        df = join_dataframes(dfs=datasets, join_on=config.join_on, how="outer_coalesce")
        splits = split_data(df, config=config)
        splits = add_labels(splits=splits, config=config)
        metadata = make_subject_metadata(splits=splits)
        metadata.write_csv(config.filepaths.data.raw.metadata)
    else:
        splits = {
            name: pl.read_csv(path)
            for name, path in config.filepaths.data.raw.splits.model_dump().items()
        }
    return splits
