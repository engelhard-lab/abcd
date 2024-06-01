from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import pandas as pd
import seaborn as sns
import seaborn.objects as so

# from shap import summary_plot
from matplotlib.patches import Patch
import textwrap

from abcd.config import Config


FORMAT = "png"


def shap_plot(shap_coefs, metadata, textwrap_width, subset: list[str] | None = None):
    n_display = 20
    plt.figure(figsize=(16, 14))
    metadata = metadata.rename(
        {"column": "variable", "dataset": "Dataset"}
    ).with_columns((pl.col("Dataset") + " " + pl.col("respondent")).alias("Dataset"))
    df = pl.read_csv("data/results/shap_coefs.csv")
    df = shap_coefs.join(other=metadata, on="variable", how="inner")
    if subset:
        df = df.filter(pl.col("Dataset").is_in(subset))
    top_questions = (
        df.group_by("question")
        .agg(pl.col("value").abs().mean())
        .sort(by="value")
        .tail(n_display)["question"]
        .reverse()
        .to_list()
    )
    df = (
        df.filter(pl.col("question").is_in(top_questions))
        .select(["question", "value", "Dataset"])
        .to_pandas()
    )
    g = sns.pointplot(
        data=df,
        x="value",
        y="question",
        hue="Dataset",
        errorbar=("sd", 2),
        linestyles="none",
        order=top_questions,
    )
    g.set(
        xlabel="SHAP value coefficient (impact of feature on model output)",
        ylabel="Feature description",
    )
    handles, labels = g.get_legend_handles_labels()
    sorted_labels, sorted_handles = zip(
        *sorted(zip(labels, handles), key=lambda t: t[0])
    )
    g.legend(sorted_handles, sorted_labels, loc="lower left", title="Dataset")
    g.autoscale(enable=True)
    g.set_yticks(g.get_yticks())
    labels = [
        textwrap.fill(label.get_text(), textwrap_width) for label in g.get_yticklabels()
    ]
    g.set_yticklabels(labels)
    g.yaxis.grid(True)
    # ax2 = g.twinx() # TODO add response
    # ax2.set_yticks(g.get_yticks())
    # right_labels = [label for label in g.get_yticklabels()]
    # ax2.set_yticklabels(right_labels)
    # ax2.set_ylabel("Response")
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    if subset:
        postfix = "_" + "_".join(subset)
    else:
        postfix = ""
    plt.savefig(f"data/plots/shap_coefs{postfix}.{FORMAT}", format=FORMAT)


def format_shap_df(df: pl.DataFrame, column_mapping, sex=None):
    sex = df.drop_in_place("Sex")[0]
    df = (
        df.transpose(include_header=True, header_name="variable")
        .with_columns(pl.col("variable").replace(column_mapping).alias("dataset"))
        .group_by("dataset")
        .sum()
        .drop("variable")
    )
    columns = df.drop_in_place("dataset")
    df = df.transpose(column_names=columns).melt().with_columns(pl.col("value").abs())
    if sex:
        df = df.with_columns(pl.lit(sex).alias("Sex"))
    return df


def grouped_shap_plot(shap_values: pl.DataFrame, column_mapping):
    plt.figure(figsize=(10, 8))
    shap_values = format_shap_df(shap_values, column_mapping)
    order = (
        shap_values.group_by("variable")
        .sum()
        .sort("value", descending=True)["variable"]
        .to_list()
    )
    g = sns.pointplot(
        data=shap_values.to_pandas(),
        x="value",
        y="variable",
        # hue="sex",
        linestyles="none",
        order=order,
        errorbar=("se", 2),
        # estimator="median",
    )
    g.set(ylabel="Feature group", xlabel="Absolute summed SHAP values")
    g.yaxis.grid(True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/plots/shap_grouped.{FORMAT}", format=FORMAT)


def sex_shap_plot(df: pl.DataFrame, column_mapping, subset=None):
    plt.figure(figsize=(10, 8))
    df = pl.read_csv("data/results/sex_shap_coefs.csv")
    df = df.group_by(
        "Sex",
    ).map_groups(lambda df: format_shap_df(df, column_mapping))
    # order = (
    #     df.group_by("variable", "Sex")
    #     .agg(pl.col("value").sum().abs())
    #     .group_by("variable")
    #     .agg(pl.col("value").diff(null_behavior="drop").abs().mean())
    # .explode("value")
    # .sort("value", descending=True)
    # ["variable"]
    # .to_list()
    # )
    # print(order)
    g = sns.pointplot(
        data=df.to_pandas(),
        x="value",
        y="variable",
        hue="Sex",
        linestyles="none",
        # order=order,
    )
    g.set(ylabel="Feature group", xlabel="Absolute summed SHAP values")
    g.yaxis.grid(True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/plots/sex_shap.{FORMAT}", format=FORMAT)


def shap_clustermap(shap_values, feature_names, column_mapping, color_mapping):
    plt.figure()
    sns.set_theme(style="white", palette="deep")
    sns.set_context("paper", font_scale=2.0)
    column_colors = {}
    color_mapping["ACEs"] = sns.color_palette("tab20")[-1]
    color_mapping["Spatiotemporal"] = sns.color_palette("tab20")[-2]
    for col, dataset in column_mapping.items():
        column_colors[col] = color_mapping[dataset]
    colors = [column_colors[col] for col in feature_names[1:] if col in column_colors]
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
    handles = [Patch(facecolor=color) for color in color_mapping.values()]
    plt.legend(
        handles,
        color_mapping.keys(),
        title="Dataset",
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(f"data/plots/shap_clustermap.{FORMAT}", format=FORMAT)


def format_shap_values(shap_values_list, X, sex):
    shap_values = np.mean(shap_values_list, axis=-1)
    shap_values = shap_values.reshape(-1, shap_values.shape[2])
    return pl.DataFrame(pd.DataFrame(shap_values, columns=X.columns)).with_columns(
        pl.lit(sex).alias("Sex")
    )


def auc_subsets():
    plt.figure(figsize=(10, 8))
    df = pl.read_csv("data/results/roc_pr.csv")
    df = df.melt(id_vars="subset")
    print(df)
    g = sns.barplot(
        data=df.to_pandas(),
        x="variable",
        y="value",
        hue="subset",
        errorbar=("sd", 2),
    )
    g.set(ylabel="AUROC", xlabel="p-factor quartile at $t+1$")
    g.legend(title="Subset at $t$")
    plt.tight_layout()
    plt.show()


def metric_curves():
    df = pl.read_csv("data/results/roc_pr.csv")
    p = sns.relplot(
        data=df,
        x="x",
        y="y",
        hue="p-factor quartile$_{t+1}$",
        style="p-factor quartile$_t$",
        col="metric",
        # row="group",
        kind="line",
        errorbar=None,
        palette="deep",
        facet_kws={"sharey": False, "sharex": False, "legend_out": True},
    )
    for i, ax in enumerate(p.axes.flat):
        if i in {0, 2}:
            ax.plot([0, 1], [0, 1], linestyle="-.", color="black")
            ax.set_ylabel("True positive rate")
            ax.set_xlabel("False positive rate")
        else:
            ax.axhline(0.25, linestyle="-.", color="black")
            ax.set_ylabel("Precision")
            ax.set_xlabel("Recall")
    p.set_titles("")
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    plt.show()


def plot(dataloader):
    sns.set_theme(style="darkgrid", palette="deep")
    sns.set_context("paper", font_scale=1.75)

    metric_curves()
    # feature_names = (
    #     pl.read_csv("data/analytic/test.csv", n_rows=1)
    #     .drop(["src_subject_id", "p_factor"])
    #     .columns
    # )
    # test_dataloader = iter(dataloader)
    # X, _ = next(test_dataloader)
    # X = pd.DataFrame(X.mean(dim=1), columns=feature_names)  # .view(-1, X.shape[2])

    # metadata = pl.read_csv("data/variables.csv")
    # shap_coefs = pl.read_csv("data/results/shap_coefs.csv")

    # shap_plot(shap_coefs=shap_coefs, metadata=metadata, textwrap_width=75)
    # shap_plot(
    #     shap_coefs=shap_coefs,
    #     metadata=metadata,
    #     subset=["DTIFA Youth", "RSFMRI Youth", "FMRI Task Youth"],
    #     textwrap_width=50,
    # )

    # column_mapping = dict(zip(metadata["column"], metadata["dataset"]))
    # shap_values = pl.read_csv("data/results/shap_values.csv")
    # male_shap_values = format_shap_values(shap_values_list, X, sex="Male")
    # shap_values_list = torch_load("data/results/shap_values_female.pt")
    # female_shap_values = format_shap_values(shap_values_list, X, sex="Female")
    # shap_values = pl.concat([male_shap_values, female_shap_values])

    # grouped_shap_plot(shap_values=shap_values, column_mapping=column_mapping)
    # sex_shap_plot(df=shap_values, column_mapping=column_mapping)

    # names = [data["name"] for data in config.features.model_dump().values()]
    # color_mapping = dict(zip(names, sns.color_palette("tab20")))
    # shap_clustermap(
    #     shap_values=shap_values,
    #     feature_names=feature_names,
    #     column_mapping=column_mapping,
    #     color_mapping=color_mapping,
    # )


if __name__ == "__main__":
    plot(dataloader=None)
    # auc_subsets()
    # roc_curves()
