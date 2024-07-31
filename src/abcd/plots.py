import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
from toml import load

# from shap import summary_plot
from matplotlib.patches import Patch
import textwrap
from abcd.config import Config


FORMAT = "pdf"


def shap_plot(shap_coefs, metadata, textwrap_width, subset: list[str] | None = None):
    n_display = 20
    plt.figure(figsize=(30, 14))
    metadata = metadata.rename({"dataset": "Dataset"}).with_columns(
        (pl.col("Dataset") + " " + pl.col("respondent")).alias("Dataset")
    )
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
    df = df.filter(pl.col("question").is_in(top_questions)).select(
        ["question", "value", "Dataset"]
    )
    sns.set_context("paper", font_scale=2.5)
    g = sns.pointplot(
        data=df,
        x="value",
        y="question",
        hue="Dataset",
        errorbar=("sd", 2),
        linestyles="none",
        order=top_questions,
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    g.set(
        xlabel="SHAP value coefficient (impact of predictor on model output)",
        ylabel="Predictor",
    )
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


def format_shap_df(df: pl.DataFrame, column_mapping):
    df = (
        df.transpose(include_header=True, header_name="variable")
        .with_columns(pl.col("variable").replace(column_mapping).alias("dataset"))
        .group_by("dataset")
        .sum()
        .drop("variable")
    )
    columns = df.drop_in_place("dataset")
    df = df.transpose(column_names=columns).melt().with_columns(pl.col("value").abs())
    return df


def grouped_shap_plot(shap_values: pl.DataFrame, column_mapping):
    plt.figure(figsize=(16, 8))
    shap_values = format_shap_df(shap_values, column_mapping)
    shap_values = shap_values.with_columns(
        pl.col("variable").replace(
            {
                "Demographics": "Age and sex",
                "Spatiotemporal": "Site and year",
            }
        )
    )
    order = (
        shap_values.group_by("variable")
        .sum()
        .sort("value", descending=True)["variable"]
        .to_list()
    )
    g = sns.pointplot(
        data=shap_values,
        x="value",
        y="variable",
        linestyles="none",
        order=order,
        errorbar=("se", 2),
    )
    g.set(ylabel="Predictor group", xlabel="Absolute summed SHAP values")
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
    g.set(ylabel="Predictor group", xlabel="Absolute summed SHAP values")
    g.yaxis.grid(True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/plots/sex_shap.{FORMAT}", format=FORMAT)


def shap_clustermap(shap_values, feature_names, column_mapping, color_mapping):
    plt.figure()
    sns.set_theme(style="white", palette="deep")
    sns.set_context("paper", font_scale=2.0)
    column_colors = {}
    color_mapping["Adverse childhood experiences (ACEs)"] = sns.color_palette("tab20")[
        -1
    ]
    color_mapping["Site and year"] = sns.color_palette("tab20")[-2]
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


def quartile_curves():
    df = pl.read_csv("data/results/curves.csv")
    df = (
        df.filter(
            (pl.col("Group").eq("{1,2,3,4}") | pl.col("Next quartile").eq(4))
            & pl.col("Variable").eq("Quartile subset")
        )
        .with_columns(pl.col("Metric").cast(pl.Enum(["ROC", "PR"])))
        .sort("Metric")
    )
    g = sns.relplot(
        data=df,
        x="x",
        y="y",
        hue="Next quartile",
        style="Curve",
        row="Metric",
        col="Group",
        kind="line",
        errorbar=None,
        palette="deep",
        facet_kws={"sharex": False, "sharey": False},
    )
    g.set_titles("")
    labels = ["a", "b", "c", "d", "e", "f"]
    for i, (ax, label) in enumerate(zip(g.axes.flat, labels)):
        ax.text(0.05, 0.95, label, fontsize=22)
        if i == 3:
            ax.set_ylabel("Positive predictive value")
        if i == 0:
            ax.set_ylabel("Sensitivity (true positive rate)")
        if i <= 2:
            ax.set_xlabel("Type I error (false positive rate)")
        else:
            ax.set_xlabel("Sensitivity (true positive rate)")
            ax.set_ylim(-0.05, 1.05)
    plt.savefig(f"data/plots/curves.{FORMAT}", format=FORMAT)


def demographic_curves():
    df = pl.read_csv("data/results/curves.csv")
    df = (
        df.filter(
            pl.col("Variable").ne("Quartile subset"),
            pl.col("Next quartile").eq(4),
            pl.col("Group").ne("16"),
        )
        .with_columns(
            pl.col("Metric").cast(pl.Enum(["ROC", "PR"])),
            pl.col("Curve").cast(pl.Enum(["Model", "Baseline"])),
            pl.when(pl.col("Curve").eq("Model") & pl.col("Metric").eq("ROC"))
            .then(pl.col("x"))
            .when(pl.col("Metric").eq("PR"))
            .then(pl.col("x"))
            .otherwise(pl.lit(None)),
        )
        .sort("Metric", "Curve", "Variable", "Group")
    )
    g = sns.FacetGrid(
        df,
        row="Variable",
        col="Metric",
        height=4,
        aspect=1.5,
        sharex=False,
        sharey=False,
    )
    for (row_var, col_var), facet_df in df.group_by("Variable", "Metric"):  # type: ignore
        ax = g.axes_dict[(row_var, col_var)]
        facet_df = facet_df.sort(pl.col("Group").cast(pl.Float32, strict=False))
        row_var = str(row_var)
        sns.lineplot(
            x="x",
            y="y",
            hue=row_var,
            style="Curve",
            data=facet_df.rename({"Group": row_var}),
            ax=ax,
            errorbar=None,
        )
        if col_var == "PR":
            ax.set_ylabel("Positive predictive value")
            ax.set_xlabel(xlabel="Sensitivity (true positive rate)")
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        else:
            ax.legend().remove()
            ax.set_ylabel("Sensitivity (true positive rate)")
            ax.set_xlabel("Type I error (false positive rate)")
            ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    g.set_titles("")
    plt.tight_layout()
    plt.savefig(
        f"data/plots/demographic_curves.{FORMAT}", format=FORMAT, bbox_inches="tight"
    )


def demographic_roc_curves():
    df = (
        pl.read_csv("data/results/demographic_roc.csv")
        .filter(
            pl.col("Label").eq(4),
            pl.col("Group").ne("16"),
            pl.col("Variable").ne("All"),
        )
        .sort("Variable", "Group")
    )
    sns.set_theme(font_scale=2.5)
    g = sns.FacetGrid(df, col="Variable", col_wrap=2, height=8)
    for col, facet_df in df.group_by(by="Variable"):
        col = str(col)
        ax = g.axes[g.col_names.index(col)]
        sns.lineplot(
            x="False positive rate",
            y="True positive rate",
            hue=col,
            # style="Curve",
            data=facet_df.rename({"Group": col}),
            ax=ax,
            errorbar=None,
        )
        ax.set(xlabel="", ylabel="")
        ax.legend(title=col, loc="lower right")
    g.set_titles("")
    g.set_axis_labels(
        "Type I error (false positive rate)", "Sensitivity (true positive rate)"
    )
    labels = ["A", "B", "C", "D"]
    for ax, label in zip(g.axes.flat, labels):
        ax.text(0.05, 0.95, label, fontsize=24)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.tight_layout()
    plt.savefig(
        f"data/plots/demographic_roc.{FORMAT}", format=FORMAT, bbox_inches="tight"
    )


def cbcl_distributions(config: Config):
    cbcl_scales = config.labels.cbcl_labels + [
        "cbcl_scr_syn_internal_t",
        "cbcl_scr_syn_external_t",
    ]
    df = pl.read_csv(
        "data/labels/mh_p_cbcl.csv",
        columns=["src_subject_id", "eventname"] + cbcl_scales,
    )
    bin_labels = [str(i) for i in range(config.preprocess.n_quantiles)]
    p_factor = (
        pl.read_csv("data/labels/p_factors.csv")
        .select(["src_subject_id", "eventname", "p_factor"])
        .with_columns(
            pl.col("p_factor")
            .qcut(
                quantiles=config.preprocess.n_quantiles,
                labels=bin_labels,
                allow_duplicates=True,
            )
            .cast(pl.Int32)
            .add(1)
        )
    )
    df = df.join(p_factor, on=["src_subject_id", "eventname"], how="inner").melt(
        id_vars=["src_subject_id", "eventname", "p_factor"]
    )
    cbcl_names = [
        "Anxious/Depressed",
        "Withdrawn/Depressed",
        "Somatic Complaints",
        "Social Problems",
        "Thought Problems",
        "Attention Problems",
        "Rule-Breaking Behavior",
        "Aggressive Behavior",
        "Internalizing",
        "Externalizing",
    ]
    name_mapping = dict(zip(cbcl_scales, cbcl_names))
    df = (
        df.with_columns(
            pl.col("variable").replace(name_mapping),
            pl.col("p_factor").cast(pl.String).alias("Quartile"),
        )
        .rename({"variable": "CBCL Syndrome Scale", "value": "t-score"})
        .sort("Quartile")
    )
    sns.set_theme(font_scale=3.0)
    g = sns.catplot(
        data=df,
        x="Quartile",
        y="t-score",
        col="CBCL Syndrome Scale",
        kind="box",
        col_wrap=4,
        height=6,
    )
    # g.tick_params(axis="x", rotation=30)
    g.set_titles("{col_name}")
    plt.savefig(
        f"data/plots/cbcl_distributions.{FORMAT}",
        format=FORMAT,
        bbox_inches="tight",
    )


def metric_comparison():
    df = pl.read_csv(
        "data/cbcl/results/metrics/metrics.csv"
    ).with_columns(  # "data/results/metrics.csv"
        pl.col("Metric").cast(pl.Enum(["AUROC", "AP"])),
        pl.col("Group").str.replace("Year ", ""),
        pl.col("Variable").str.replace("Measurement year", "Year"),
    )
    metrics_df1 = df.filter(pl.col("Variable").eq("Quartile subset")).with_columns(
        pl.lit("Autoregressive").alias("Feature set")
    )
    # print(metrics_table+)
    df = pl.read_csv("data/results/metrics.csv").with_columns(  #
        pl.col("Metric").cast(pl.Enum(["AUROC", "AP"])),
        pl.col("Group").str.replace("Year ", ""),
        pl.col("Variable").str.replace("Measurement year", "Year"),
    )
    metrics_df2 = df.filter(pl.col("Variable").eq("Quartile subset")).with_columns(
        pl.lit("Questions + brain").alias("Feature set")
    )
    metrics_df = pl.concat([metrics_df1, metrics_df2]).select(
        pl.col("Feature set"), pl.exclude("Feature set")
    )
    print(metrics_df)
    g = sns.catplot(
        data=metrics_df.to_pandas(),
        x="Group",
        y="value",
        hue="Feature set",
        col="Next quartile",
        row="Metric",
        kind="bar",
    )
    g.set_titles("{row_name}: Q{col_name}")
    plt.show()


def plot(config):
    sns.set_theme(style="darkgrid", palette="deep", font_scale=2.0)
    sns.set_context("paper", font_scale=2.0)

    metric_comparison()

    # cbcl_distributions(config=config)
    # quartile_curves()
    # demographic_curves()
    # feature_names = (
    #     pl.read_csv("data/analytic/test.csv", n_rows=1)
    #     .drop(["src_subject_id", "p_factor", "label"])
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

    # column_mapping = dict(zip(metadata["variable"], metadata["dataset"]))
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
    config = Config(**load("config.toml"))
    plot(config)
