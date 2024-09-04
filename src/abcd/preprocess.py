from collections import defaultdict
from typing import Callable
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.decomposition import FactorAnalysis
import polars as pl
import polars.selectors as cs

from functools import partial, reduce

from abcd.config import Config

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


def join_dataframes(dfs: list[pl.DataFrame], join_on: list[str]) -> pl.DataFrame:
    return reduce(
        lambda left, right: left.join(
            right,
            how="full",
            coalesce=True,
            on=join_on,
        ),
        dfs,
    ).sort(join_on)


def make_demographics(df: pl.DataFrame):
    education = ["demo_prnt_ed_v2", "demo_prtnr_ed_v2"]
    sex = ["demo_sex_v2_null", "demo_sex_v2_2", "demo_sex_v2_3"]
    join_on = ["src_subject_id", "eventname"]
    df = (
        df.with_columns(pl.max_horizontal(education).alias("parent_highest_education"))
        .with_columns(pl.all().forward_fill().over("src_subject_id"))
        .drop(education)
        .to_dummies("demo_sex_v2")
        .select(cs.exclude(education + sex))
        .sort(join_on)
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
        ).select(*join_on, "adi_percentile")
        # .drop(adi_columns)
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
            .select(~cs.contains(*columns_to_drop))
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


def filter_data(df: pl.DataFrame, label_columns: list[str], group: str):
    return df.filter(~pl.all_horizontal(pl.col(label_columns).is_null())).with_columns(
        pl.all().forward_fill().over(group)
    )


def impute_within_subject(df: pl.DataFrame):
    return df.with_columns(pl.all().forward_fill().over("src_subject_id"))


def generate_data(config: Config):
    datasets = get_datasets(config=config)
    return (
        join_dataframes(dfs=datasets, join_on=config.join_on)
        .pipe(
            filter_data,
            label_columns=config.features.mh_p_cbcl.columns,
            group=config.join_on[0],
        )
        .pipe(impute_within_subject)
    )


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


def get_features(df: pl.DataFrame, analysis: str, config: Config):
    brain_features = get_brain_features(config)
    exclude = [config.join_on[0], "race_ethnicity", "interview_date"]
    match analysis:
        case "metadata":
            df = df.select(cs.exclude(exclude[:-2]))
        case "questions_mri":
            df = df.select(cs.exclude(exclude + config.features.mh_p_cbcl.columns))
        case "questions":
            df = df.select(
                cs.exclude(exclude + config.features.mh_p_cbcl.columns + brain_features)
            )
        case "questions_symptoms":
            df = df.select(cs.exclude(exclude + brain_features))
        case "symptoms" | "autoregressive":
            df = df.select(config.features.mh_p_cbcl.columns)
        case "questions_mri_symptoms":
            df = df.select(cs.exclude(exclude))
        case _:
            raise ValueError(f"Invalid analysis: {analysis}")
    return df.columns


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
    return {"train": pl.concat(train), "val": pl.concat(val), "test": pl.concat(test)}


def format_labels(
    splits: dict[str, pl.DataFrame], group: str, analysis: str
) -> dict[str, pl.DataFrame]:
    shift_y = pl.col("y_t").shift(-1).over(group).alias("y_{t+1}")
    columns = (group, "y_t", "y_{t+1}")
    for name, split in splits.items():
        split = (
            split.with_columns(shift_y)
            .select(pl.col(columns), pl.exclude(columns))
            .drop_nulls(subset=["y_{t+1}"])
        )
        if analysis != "metadata":
            split = split.drop("y_t")
        if analysis == "autoregressive":
            split = split.select(group, "p-factor", "y_{t+1}")
        splits[name] = split
    return splits


def make_transformer(
    analysis: str,
    group: str,
    feature_columns: list[str],
    label_columns: list[str],
    n_quantiles: int,
):
    identity_transformer = FunctionTransformer(lambda x: x)
    if analysis == "metadata":
        feature_pipeline = identity_transformer
    elif analysis == "autoregressive":
        feature_pipeline = make_pipeline(
            StandardScaler(),
            FactorAnalysis(n_components=1),
            FunctionTransformer(lambda x: x.rename({"factoranalysis0": "p-factor"})),
        )
    else:
        feature_pipeline = make_pipeline(StandardScaler())
    label_pipeline = make_pipeline(
        StandardScaler(),
        FactorAnalysis(n_components=1),
        KBinsDiscretizer(n_bins=n_quantiles, encode="ordinal", strategy="quantile"),
        FunctionTransformer(lambda x: x.rename({"factoranalysis0": "y_t"})),
    )
    transformers = [
        ("indices", identity_transformer, [group]),
        ("features", feature_pipeline, feature_columns),
        ("labels", label_pipeline, label_columns),
    ]
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def impute_by_time_point(df: pl.DataFrame):
    return df.with_columns(cs.numeric().fill_null(strategy="mean").over("eventname"))


def transform_within_event(
    splits: dict[str, pl.DataFrame], transformer: Callable, event: str
):
    transformers = {}
    train_groups = []
    for name, train_group in splits["train"].group_by(event, maintain_order=True):
        transformers[name] = transformer()
        train_groups.append(transformers[name].fit_transform(train_group))
    val_groups = []
    for name, val_group in splits["val"].group_by(event, maintain_order=True):
        val_groups.append(transformers[name].transform(val_group))
    test_groups = []
    for name, test_group in splits["test"].group_by(event, maintain_order=True):
        test_groups.append(transformers[name].transform(test_group))
    splits["train"] = pl.concat(train_groups)
    splits["val"] = pl.concat(val_groups)
    splits["test"] = pl.concat(test_groups)
    return splits


def transform_data(
    splits: dict,
    group: str,
    event: str,
    feature_columns: list[str],
    label_columns: list[str],
    analysis: str,
    factor_model: str,
    n_quantiles: int,
) -> dict[str, pl.DataFrame]:
    partial_transformer = partial(
        make_transformer,
        analysis=analysis,
        group=group,
        feature_columns=feature_columns,
        label_columns=label_columns,
        n_quantiles=n_quantiles,
    )
    for name, split in splits.items():
        splits[name] = split.pipe(impute_by_time_point)
    if factor_model == "within_event":
        splits = transform_within_event(
            splits=splits,
            transformer=partial_transformer,
            event=event,
        )
    else:
        transformer = partial_transformer()
        splits["train"] = transformer.fit_transform(splits["train"])
        splits["val"] = transformer.transform(splits["val"])
        splits["test"] = transformer.transform(splits["test"])
    splits = format_labels(splits=splits, group=group, analysis=analysis)
    return splits


def make_dataset(config: Config, analysis: str, factor_model: str):
    df = generate_data(config=config)
    df.write_csv(config.filepaths.data.raw.dataset)
    group, event = config.join_on
    splits = split_data(
        df=df,
        group=config.join_on[0],
        train_size=config.preprocess.train_size,
        random_state=config.random_seed,
    )
    features = get_features(df, analysis=analysis, config=config)
    splits = transform_data(
        splits=splits,
        group=group,
        event=event,
        feature_columns=features,
        label_columns=config.features.mh_p_cbcl.columns,
        analysis=analysis,
        factor_model=factor_model,
        n_quantiles=config.preprocess.n_quantiles,
    )
    splits["train"].write_csv(config.filepaths.data.analytic.train)
    splits["val"].write_csv(config.filepaths.data.analytic.val)
    splits["test"].write_csv(config.filepaths.data.analytic.test)
    return splits


def get_dataset(analysis: str, factor_model: str, config: Config):
    if config.regenerate:
        splits = make_dataset(config, analysis, factor_model)
    else:
        splits = {
            "train": pl.read_csv(config.filepaths.data.analytic.train),
            "val": pl.read_csv(config.filepaths.data.analytic.val),
            "test": pl.read_csv(config.filepaths.data.analytic.test),
        }
    return splits
