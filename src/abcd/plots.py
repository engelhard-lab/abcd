import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from toml import load

# from shap import summary_plot
from matplotlib.patches import Patch
import textwrap
from abcd.config import Config


FORMAT = "pdf"


def shap_plot(analysis, factor_model, metadata, textwrap_width):
    n_display = 20
    plt.figure(figsize=(30, 16))
    metadata = metadata.rename({"dataset": "Dataset"})
    # .with_columns(
    #     (pl.col("Dataset") + " " + pl.col("respondent")).alias("Dataset")
    # )
    shap_coefs = pl.read_csv(
        f"data/analyses/{factor_model}/{analysis}/results/shap_coefficients.csv"
    )
    df = shap_coefs.join(other=metadata, on="variable", how="inner").with_columns(
        (pl.col("respondent") + ": " + pl.col("Dataset")).alias("Dataset")
    )
    top_questions = (
        df.group_by("respondent", "question")
        .agg(pl.col("value").sum().abs())
        .sort(by="value")
        .tail(n_display)["question"]
        .reverse()
        .to_list()
    )
    df = df.filter(pl.col("question").is_in(top_questions)).select(
        ["question", "value", "Dataset", "respondent"]
    )
    sns.set_context("paper", font_scale=2.7)
    g = sns.pointplot(
        data=df.to_pandas(),
        x="value",
        y="question",
        hue="Dataset",
        errorbar="pi",
        linestyles="none",
        order=top_questions,
    )
    # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    g.set(
        xlabel="SHAP value coefficient",
        ylabel="Predictor",
    )
    # g.autoscale(enable=True)
    g.set_yticks(g.get_yticks())
    labels = [
        textwrap.fill(label.get_text(), textwrap_width) for label in g.get_yticklabels()
    ]
    g.set_yticklabels(labels)
    g.yaxis.grid(True)

    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/supplement/plots/supplemental_figure_3.{FORMAT}", format=FORMAT)


def grouped_shap_plot(shap_values: pl.DataFrame, metadata: pl.DataFrame):
    plt.figure(figsize=(16, 8))
    shap_values = (
        shap_values.unpivot()
        .join(metadata, on="variable", how="inner")
        .rename({"respondent": "Respondent"})
    )
    n_bootstraps = 500
    bootstraps = []
    for boot in range(n_bootstraps):
        resampled = resample(shap_values)
        resampled = resampled.with_columns(pl.lit(boot).alias("boot"))  # type: ignore
        bootstraps.append(resampled)
    shap_values = pl.concat(bootstraps)
    shap_values = shap_values.group_by("dataset", "Respondent", "boot").sum()
    order = (
        shap_values.group_by("dataset", "Respondent", "boot")
        .agg(pl.col("value").sum().abs())
        .sort("value", descending=True)["dataset"]
        .to_list()
    )
    g = sns.pointplot(
        data=shap_values.to_pandas(),
        x="value",
        y="dataset",
        hue="Respondent",
        linestyles="none",
        order=order,
        errorbar="pi",
    )
    sns.move_legend(g, "lower right")
    g.set(ylabel="Predictor group", xlabel="Summed SHAP values")
    g.yaxis.grid(True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/plots/figure_3.{FORMAT}", format=FORMAT)


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
    df = pl.read_parquet("data/results/metrics/curves.parquet")
    df = (
        df.filter(
            pl.col("Quartile at t+1").eq(4)
            & pl.col("Variable").eq("Quartile subset")
            & pl.col("Predictor set").is_in(["{CBCL scales}", "{Questions}"])
            & pl.col("Factor model").eq("Within-event")
            & pl.col("y").ne(0)
        )
        .drop("Factor model", "Variable")
        .with_columns(
            pl.col("Metric").cast(pl.Enum(["ROC", "PR"])),
            pl.col("Group")
            .replace(
                {
                    "{1,2,3}": "High-risk conversion",
                    "{4}": "High-risk persistence",
                    "{1,2,3,4}": "High-risk",
                }  # TODO change this at the source
            )
            .cast(
                pl.Enum(
                    [
                        "High-risk conversion",
                        "High-risk persistence",
                        "High-risk",
                    ]
                )
            ),
            pl.col("Predictor set").replace(
                {
                    "{CBCL scales}": "CBCL scales",
                    "{Questions}": "Questions",
                }  # TODO change this at the source
            ),
        )
        .sort("Predictor set", "Metric", "Group", "y")
    )
    g = sns.relplot(
        data=df.to_pandas(),
        x="x",
        y="y",
        hue="Predictor set",
        row="Metric",
        col="Group",
        kind="line",
        errorbar=None,
        palette="deep",
        facet_kws={"sharex": False, "sharey": False},
    )
    g.set_titles("{col_name} {row_name} curve")
    font_size = 24
    labels = ["a", "b", "c", "d", "e", "f"]
    grouped_df = df.partition_by(["Metric", "Group"], maintain_order=True)
    for i, (ax, label, group) in enumerate(zip(g.axes.flat, labels, grouped_df)):
        if i == 3:
            ax.set_ylabel("Precision (positive predictive value)")
        if i == 0:
            ax.set_ylabel("True positive rate (sensitivity)")
        if i <= 2:
            ax.text(0.05, 0.95, label, fontsize=font_size)
            ax.set_xlabel("False positive rate (Type I error)")
            ax.plot([0, 1], [0, 1], linestyle="--", color="black")
        else:
            ax.text(0.95, 0.95, label, fontsize=font_size)
            ax.set_xlabel("Recall (sensitivity)")
            ax.axhline(
                group.filter(pl.col("y").ne(0))["y"].min(),
                linestyle="--",
                color="black",
            )
        ax.set_ylim(0.0, 1.05)
    plt.subplots_adjust(hspace=0.3)  # , wspace=0.4
    plt.savefig(f"data/plots/figure_2.{FORMAT}", format=FORMAT)


def cbcl_distributions(config: Config):
    cbcl_scales = config.features.mh_p_cbcl.columns
    # + [
    #     "cbcl_scr_syn_internal_t",
    #     "cbcl_scr_syn_external_t",
    # ]
    cbcl_names = [
        "Anxious/Depressed",
        "Withdrawn/Depressed",
        "Somatic Complaints",
        "Social Problems",
        "Thought Problems",
        "Attention Problems",
        "Rule-Breaking Behavior",
        "Aggressive Behavior",
        # "Internalizing",
        # "Externalizing",
    ]
    name_mapping = dict(zip(cbcl_scales, cbcl_names))
    df = (
        pl.read_csv(
            "data/analyses/within_event/symptoms/analytic/train.csv",
            columns=["src_subject_id", "y_{t+1}"] + cbcl_scales,
        )
        .rename({"y_{t+1}": "p-factor quartile"})
        .unpivot(index="p-factor quartile", on=cbcl_scales)
        .rename({"variable": "CBCL scale", "value": "t-score"})
        .with_columns(
            pl.col("CBCL scale").replace(name_mapping),
            pl.col("p-factor quartile").add(1).cast(pl.Int32),
        )
    )
    sns.set_theme(font_scale=2.0)
    g = sns.catplot(
        data=df.to_pandas(),
        x="p-factor quartile",
        y="t-score",
        col="CBCL scale",
        kind="boxen",
        col_wrap=4,
        height=6,
    )
    g.set_titles("{col_name}")
    plt.savefig(
        f"data/plots/cbcl_distributions.{FORMAT}",
        format=FORMAT,
        bbox_inches="tight",
    )


def analysis_comparison():
    df = pl.read_parquet("data/results/metrics/metrics.parquet")
    df = df.filter(
        pl.col("Variable").eq("Quartile subset")
        & pl.col("Metric").eq("AUROC")
        & pl.col("Factor model").eq("Within-event")
    )
    g = sns.catplot(
        data=df.to_pandas(),
        x="Quartile at t+1",
        y="value",
        kind="bar",
        hue="Predictor set",
        col="Group",
        errorbar="pi",
    )
    g.set(ylim=(0.5, 1.0))
    g.set_axis_labels("Quartile at t+1", "AUROC")
    for ax in g.axes.flat:
        ax.axhline(0.5, color="black", linestyle="--")
    plt.savefig(
        f"data/supplement/plots/supplemental_figure_2.{FORMAT}",
        format=FORMAT,
        bbox_inches="tight",
    )


def p_factor_model_comparison():
    df = pl.read_parquet("data/results/metrics/metrics.parquet")
    df = df.filter(
        pl.col("Predictor set").is_in(["{Questions}", "{CBCL scales}"])
        & pl.col("Group").eq("{1,2,3}")
        & pl.col("Metric").eq("AUROC")
    )
    g = sns.catplot(
        data=df.to_pandas(),
        x="Quartile at t+1",
        y="value",
        kind="bar",
        hue="Factor model",
        col="Predictor set",
        errorbar="pi",
    )
    g.set(ylim=(0.45, 1.0))
    g.set_axis_labels("Quartile at t+1", "AUROC")
    for ax in g.axes.flat:
        ax.axhline(0.5, color="black", linestyle="--")
    plt.savefig(
        f"data/supplement/plots/supplemental_figure_3.{FORMAT}",
        format=FORMAT,
        bbox_inches="tight",
    )


def plot(config):
    sns.set_theme(style="darkgrid", palette="deep", font_scale=2.0)
    sns.set_context("paper", font_scale=2.0)

    # analysis_comparison()
    quartile_curves()
    # p_factor_model_comparison()

    # cbcl_distributions(config=config)
    # demographic_curves()

    # metadata = (
    #     pl.read_csv("data/supplement/files/supplemental_file_1.csv").filter(
    #         pl.col("dataset").ne("Site and measurement")
    #     )
    # .with_columns(
    #     pl.col("dataset").replace({"Site and measurement": "Follow-up event"})
    # )
    # )  # FIXME file_1.csv -> table_1.csv
    # analysis = "questions_mri"
    # factor_model = "within_event"
    # shap_plot(
    #     analysis=analysis,
    #     factor_model=factor_model,
    #     metadata=metadata,
    #     textwrap_width=80,
    # )
    # shap_plot(analysis="symptoms", metadata=metadata, textwrap_width=75)
    # shap_plot(analysis="questions_symptoms", metadata=metadata, textwrap_width=75)

    # shap_plot(
    #     shap_coefs=shap_coefs,
    #     metadata=metadata,
    #     subset=["DTIFA Youth", "RSFMRI Youth", "FMRI Task Youth"],
    #     textwrap_width=50,
    # )
    # column_mapping = dict(zip(metadata["variable"], metadata["dataset"]))
    # shap_values = pl.read_csv(
    #     f"data/analyses/{factor_model}/{analysis}/results/shap_values.csv"
    # )
    # grouped_shap_plot(shap_values=shap_values, metadata=metadata)
    # male_shap_values = format_shap_values(shap_values_list, X, sex="Male")
    # shap_values_list = torch_load("data/results/shap_values_female.pt")
    # female_shap_values = format_shap_values(shap_values_list, X, sex="Female")
    # shap_values = pl.concat([male_shap_values, female_shap_values])

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
