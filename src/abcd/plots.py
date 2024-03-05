import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from torch import load as torch_load
from shap import summary_plot
from matplotlib.patches import Patch

from abcd.dataset import RNNDataset, collate
from abcd.evaluate import RACE_MAPPING, SEX_MAPPING
from abcd.preprocess import DATASET_MAPPING


FORMAT = "png"

DATASET_NAMES = [
    "family_environment",
    "neighborhood",
    "problem_monitor",
    "prosocial",
    "school",
    "screentime",
    "sleep_disturbance",
    "rules",
    "brain_dti_fa",
    "brain_rsfmri",
    "brain_sst",
]
COLOR_MAPPING = dict(
    zip(DATASET_NAMES, sns.color_palette("tab20"))
)  # FIXME get better colors


def predicted_vs_observed():
    plt.figure()
    sns.set_palette(palette=sns.color_palette("tab20"))
    df = (
        pl.read_csv("data/results/predicted_vs_observed.csv")
        .with_columns(
            (pl.col("age") / 12).round(0).cast(pl.Int32),
            pl.col("race_ethnicity").replace(RACE_MAPPING),
            pl.col("sex").replace(SEX_MAPPING),
        )
        .rename(
            {
                "eventname": "Year",
                "race_ethnicity": "Race/Ethnicity",
                "sex": "Sex",
                "age": "Age",
            }
        )
    )
    df = (
        df.melt(
            id_vars=["y_pred", "y_true"],
            value_vars=["Year", "Race/Ethnicity", "Sex", "Age"],
        )
        .rename({"value": "Group", "variable": "Variable"})
        .sort("Group")
    )
    min_val = df.select(pl.min_horizontal(["y_true", "y_pred"]).min()).item()
    max_val = df.select(pl.max_horizontal(["y_true", "y_pred"]).max()).item()
    df = df.to_pandas()
    g = sns.lmplot(
        data=df,
        x="y_true",
        y="y_pred",
        hue="Group",
        col="Variable",
        col_wrap=2,
        facet_kws={"legend_out": False},
    )
    g.set(xlabel="Observed", ylabel="Predicted")
    for ax in g.axes.flat:
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            title=ax.get_title().replace("Variable = ", ""),
            loc="upper left",
        )
        ax.set_title("")
    plt.show()
    plt.savefig(f"data/plots/predicted_vs_observed.{FORMAT}", format=FORMAT)


def shap_plot(shap_values, X, feature_names):
    pass


def shap_by_year_plot(shap_values_list, X, feature_names):
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    for i, (shap_values, ax) in enumerate(zip(shap_values_list, axes.flatten())):
        shap_values = shap_values.reshape(-1, shap_values.shape[2])
        plt.sca(ax)
        ax.set_title(f"Year {i + 1}")
        summary_plot(
            shap_values,
            features=X,
            feature_names=feature_names,
            show=False,
            plot_size=None,
        )
    plt.tight_layout()
    plt.show()


def make_column_mapping(join_on):
    columns = {
        DATASET_MAPPING[dataset]: pl.read_csv(
            "data/features/" + dataset + ".csv",
            null_values="",
            infer_schema_length=50_000,
        )
        .drop(join_on)
        .columns
        for dataset in DATASET_MAPPING
    }
    return {
        value: key.replace("_youth", "").replace("_parent", "")
        for key, values in columns.items()
        for value in values
    }


def grouped_shap_plot(shap_values, feature_names, column_mapping):
    df = pl.DataFrame(shap_values, schema=feature_names).transpose(
        include_header=True, header_name="variable"
    )
    df = (
        df.filter(pl.col("variable") != "eventname")
        .with_columns(pl.col("variable").replace(column_mapping).alias("dataset"))
        .group_by("dataset")
        .sum()
        .drop("variable")
    )
    columns = df.drop_in_place("dataset")
    df = df.transpose(column_names=columns).to_numpy()
    summary_plot(df, feature_names=columns.to_list(), show=True)


def shap_clustermap(shap_values, feature_names, column_mapping):
    column_colors = {
        col: COLOR_MAPPING[dataset] for col, dataset in column_mapping.items()
    }
    colors = [column_colors[col] for col in feature_names[1:]]
    shap_df = pl.DataFrame(shap_values[:, 1:], schema=feature_names[1:]).to_pandas()
    shap_corr = shap_df.corr()
    g = sns.clustermap(
        shap_corr,
        row_colors=colors,
        yticklabels=False,
        xticklabels=False,
    )
    g.ax_col_dendrogram.set_visible(False)
    mask = np.triu(np.ones_like(shap_corr))
    values = g.ax_heatmap.collections[0].get_array().reshape(shap_corr.shape)  # type: ignore
    new_values = np.ma.array(values, mask=mask)
    g.ax_heatmap.collections[0].set_array(new_values)
    handles = [Patch(facecolor=color) for color in COLOR_MAPPING.values()]
    plt.legend(
        handles,
        COLOR_MAPPING.keys(),
        title="Dataset",
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc="upper right",
    )
    plt.show()


def plot(dataloader):
    sns.set_theme(font_scale=1.5, style="whitegrid", palette="deep")
    feature_names = (
        pl.read_csv("data/analytic/test.csv")
        .drop(["src_subject_id", "p_score"])
        .columns
    )
    test_dataloader = iter(dataloader)
    X, _ = next(test_dataloader)
    X = X.view(-1, X.shape[2])
    shap_values_list = torch_load("data/results/shap_values.pt")
    shap_values = np.mean(shap_values_list, axis=0)
    shap_values = shap_values.reshape(-1, shap_values.shape[2])
    assert X.shape[1] == len(feature_names)
    summary_plot(
        shap_values,
        features=X,
        show=True,
    )
    shap_by_year_plot(
        shap_values_list=shap_values_list, X=X, feature_names=feature_names
    )
    column_mapping = make_column_mapping(join_on=["src_subject_id", "eventname"])
    grouped_shap_plot(
        shap_values=shap_values,
        feature_names=feature_names,
        column_mapping=column_mapping,
    )
    shap_clustermap(
        shap_values=shap_values,
        feature_names=feature_names,
        column_mapping=column_mapping,
    )
    predicted_vs_observed()


if __name__ == "__main__":
    test = pd.read_csv("data/analytic/test.csv")
    test_dataset = RNNDataset(dataset=test, target="p_score")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=500,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )
    plot(dataloader=test_dataloader)
