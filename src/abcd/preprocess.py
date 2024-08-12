from collections import defaultdict
from typing import Callable
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.decomposition import FactorAnalysis
import polars as pl
from polars.type_aliases import JoinStrategy
import polars.selectors as cs

from functools import partial, reduce

from abcd.config import Config
from abcd.metadata import make_variable_metadata


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
    columns_to_drop = [config.join_on[0], "y_t", "y_{t+1}", "race_ethnicity"]
    if analysis == "metadata":
        columns_to_drop = columns_to_drop[:-1]
    match analysis:
        case "with_brain" | "by_year":
            features = cs.exclude(config.features.mh_p_cbcl.columns)
        case "without_brain":
            brain_features = get_brain_features(config)
            features = cs.exclude(config.features.mh_p_cbcl.columns + brain_features)
        case "symptoms":
            features = cs.by_name(["eventname"] + config.features.mh_p_cbcl.columns)
        case "all":
            features = cs.all()
        case _:
            raise ValueError(f"Invalid analysis: {analysis}")
    return features & cs.exclude(columns_to_drop)


def split_data(df: pl.DataFrame, group: str, train_size: float, random_state: int):
    dfs = df.partition_by(group, maintain_order=True)
    train, val_test = train_test_split(
        dfs,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )
    val, test = train_test_split(
        val_test, train_size=0.5, random_state=random_state, shuffle=False
    )
    return pl.concat(train), pl.concat(val), pl.concat(test)


def transform_data(
    splits: tuple[pl.DataFrame, ...],
    group: str,
    feature_columns: list[str],
    label_columns: list[str],
) -> tuple[pl.DataFrame, ...]:
    feature_pipeline = make_pipeline(StandardScaler(), SimpleImputer(strategy="mean"))
    label_pipeline = make_pipeline(
        StandardScaler(),
        SimpleImputer(strategy="mean"),
        FactorAnalysis(n_components=1),
        KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile"),
    )
    identity_transformer = FunctionTransformer(lambda x: x)
    transformers = [
        ("indices", identity_transformer, [group]),
        ("features", feature_pipeline, feature_columns),
        ("labels", label_pipeline, label_columns),
    ]
    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    train, val, test = splits
    train = column_transformer.fit_transform(train)  # type: ignore
    val = column_transformer.transform(val)  # type: ignore
    test = column_transformer.transform(test)  # type: ignore
    return train, val, test  # type: ignore # , column_transformer.named_transformers_["features"]


def format_labels(
    splits: tuple[pl.DataFrame, ...], group: str, observation: str
) -> tuple[pl.DataFrame, ...]:
    shift_y = pl.col("y_t").shift(-1).over(group)
    columns = (group, observation, "y_t", "y_{t+1}")
    return tuple(
        split.rename({"factoranalysis0": "y_t"})
        .with_columns(shift_y.alias("y_{t+1}"))
        .select(pl.col(columns), pl.exclude(columns))
        .drop_nulls(subset=["y_{t+1}"])
        for split in splits
    )


def get_dataset(
    splits: tuple[pl.DataFrame, ...], feature_columns: list[str], config: Config
):
    group, observation = config.join_on
    if config.regenerate:
        splits = transform_data(
            splits=splits,
            group=group,
            feature_columns=feature_columns,
            label_columns=config.features.mh_p_cbcl.columns,
        )
        splits = format_labels(splits=splits, group=group, observation=observation)
        split_dict = dict(zip(["train", "val", "test"], splits))
    else:
        split_dict = {
            name: pl.read_csv(path)
            for name, path in config.filepaths.data.raw.splits.model_dump().items()
        }
    return split_dict


def filter_data(df: pl.DataFrame, label_columns: list[str], group: str):
    return df.filter(~pl.all_horizontal(pl.col(label_columns).is_null())).with_columns(
        pl.all().forward_fill().over(group)
    )


def forward_fill_nulls(df: pl.DataFrame):
    return df.with_columns(pl.all().forward_fill().over("src_subject_id"))


def generate_data(config: Config):
    datasets = get_datasets(config=config)
    make_variable_metadata(dfs=datasets, features=config.features)
    df = (
        join_dataframes(dfs=datasets, join_on=config.join_on, how="outer_coalesce")
        .pipe(
            filter_data,
            label_columns=config.features.mh_p_cbcl.columns,
            group=config.join_on[0],
        )
        .pipe(forward_fill_nulls)
    )
    df.write_csv(config.filepaths.data.raw.dataset)
