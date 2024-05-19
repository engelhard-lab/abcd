import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import pandas as pd
import seaborn as sns

# from shap import summary_plot
from matplotlib.patches import Patch
import textwrap

from abcd.config import Config


FORMAT = "png"


def predicted_vs_observed_plot():
    df = pl.read_csv("data/results/predicted_vs_observed.csv")
    df = df.rename(
        {
            "eventname": "Year",
            "race_ethnicity": "Race/Ethnicity",
            "demo_sex_v2_1": "Sex",
            "interview_age": "Age",
        }
    )
    df = (
        df.melt(
            id_vars=["y_pred", "y_true"],
            value_vars=["Year", "Race/Ethnicity", "Sex", "Age"],
        )
        .rename({"value": "Group", "variable": "Variable"})
        .with_columns(pl.col("Group").replace("9", " 9"))
        .sort("Group")
        .drop_nulls()
    )
    min_val = df.select(pl.min_horizontal(["y_true", "y_pred"]).min()).item()
    max_val = df.select(pl.max_horizontal(["y_true", "y_pred"]).max()).item()
    df = df.to_pandas()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    groups = df.groupby("Variable")
    palette = sns.color_palette("deep")
    for j, (ax, (name, group)) in enumerate(zip(axes.flat, groups)):
        labels = group["Group"].unique().tolist()
        for i, hue_category in enumerate(labels):
            hue_subset = group[group["Group"] == hue_category]
            sns.barplot(x="Group", y="R2", data=hue_subset, color=palette[i], ax=ax)
        handles = [Patch(facecolor=palette[i]) for i in range(len(labels))]
        ax.legend(
            handles=handles,
            labels=labels,
            title=name,
            loc="upper left",
        )
        if j in (0, 2):
            ax.set_ylabel("Predicted")
        if j in (2, 3):
            ax.set_xlabel("Observed")
        if j in (1, 3):
            ax.set_ylabel("")
        if j in (0, 1):
            ax.set_xlabel("")
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/plots/predicted_vs_observed.{FORMAT}", format=FORMAT)


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


def format_shap_df(shap_values: pl.DataFrame, column_mapping):
    df = shap_values.transpose(include_header=True, header_name="variable")
    df = (
        df.with_columns(pl.col("variable").replace(column_mapping).alias("dataset"))
        .group_by("dataset")
        .sum()
        .drop("variable")
    )
    columns = df.drop_in_place("dataset")
    return df.transpose(column_names=columns).melt().with_columns(pl.col("value").abs())


def grouped_shap_plot(shap_values: pl.DataFrame, column_mapping):
    plt.figure(figsize=(10, 8))
    shap_values = format_shap_df(shap_values, column_mapping)
    order = (
        shap_values.group_by("variable")
        .mean()
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
        errorbar=("ci", 2),
        estimator="median",
    )
    g.set(ylabel="Feature group", xlabel="Absolute summed SHAP values")
    g.yaxis.grid(True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/plots/shap_grouped.{FORMAT}", format=FORMAT)


def sex_shap_plot(df: pl.DataFrame, column_mapping, subset=None):
    plt.figure(figsize=(10, 8))
    males = df.filter(pl.col("Sex").eq("Male")).drop("Sex")
    females = df.filter(pl.col("Sex").eq("Female")).drop("Sex")
    males = format_shap_df(males, column_mapping).with_columns(
        pl.lit("Male").alias("Sex")
    )
    females = format_shap_df(females, column_mapping).with_columns(
        pl.lit("Female").alias("Sex")
    )
    df = pl.concat([males, females])
    order = (
        df.group_by("variable", "Sex")
        .agg(pl.col("value").sum().abs())
        .group_by("variable")
        .agg(pl.col("value").diff(null_behavior="drop").abs())
        .explode("value")
        .sort("value", descending=True)["variable"]
        .to_list()
    )
    g = sns.pointplot(
        data=df.to_pandas(),
        x="value",
        y="variable",
        hue="Sex",
        linestyles="none",
        order=order,
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


def r2_plot():
    column_mapping = {
        "cbcl_scr_syn_anxdep_t": "Anxiety/Depression",
        "cbcl_scr_syn_withdep_t": "Withdrawn/Depressed",
        "cbcl_scr_syn_somatic_t": "Somatic Problems",
        "cbcl_scr_syn_social_t": "Social Problems",
        "cbcl_scr_syn_thought_t": "Thought Problems",
        "cbcl_scr_syn_attention_t": "Attention Problems",
        "cbcl_scr_syn_rulebreak_t": "Rule-Breaking Behavior",
        "cbcl_scr_syn_aggressive_t": "Aggressive Behavior",
    }
    df = pl.read_csv("data/results/r2_cbcl.csv")
    df = df.rename({"variable": "Label", "value": "R$^2$"}).with_columns(
        pl.col("Label").replace(column_mapping)
    )
    sns.barplot(data=df.to_pandas(), x="R$^2$", y="Label", errorbar=("sd", 2))
    plt.tight_layout()
    plt.show()


def format_shap_values(shap_values_list, X, sex):
    shap_values = np.mean(shap_values_list, axis=-1)
    shap_values = shap_values.reshape(-1, shap_values.shape[2])
    return pl.DataFrame(pd.DataFrame(shap_values, columns=X.columns)).with_columns(
        pl.lit(sex).alias("Sex")
    )


def plot(config: Config, dataloader):
    sns.set_theme(style="darkgrid", palette="deep")
    sns.set_context("paper", font_scale=1.75)
    # r2_plot()
    # predicted_vs_observed_plot()
    # names = [data["name"] for data in config.features.model_dump().values()]
    feature_names = (
        pl.read_csv("data/analytic/test.csv", n_rows=1)
        .drop(["src_subject_id", "p_factor"])
        .columns
    )
    test_dataloader = iter(dataloader)
    X, _ = next(test_dataloader)
    X = pd.DataFrame(X.mean(dim=1), columns=feature_names)  # .view(-1, X.shape[2])

    metadata = pl.read_csv("data/variables.csv")
    shap_coefs = pl.read_csv("data/results/shap_coefs.csv")

    shap_plot(shap_coefs=shap_coefs, metadata=metadata, textwrap_width=75)
    shap_plot(
        shap_coefs=shap_coefs,
        metadata=metadata,
        subset=["DTIFA Youth", "RSFMRI Youth", "FMRI Task Youth"],
        textwrap_width=50,
    )

    column_mapping = dict(zip(metadata["column"], metadata["dataset"]))
    shap_values = pl.read_csv("data/results/shap_values.csv")
    # male_shap_values = format_shap_values(shap_values_list, X, sex="Male")
    # shap_values_list = torch_load("data/results/shap_values_female.pt")
    # female_shap_values = format_shap_values(shap_values_list, X, sex="Female")
    # shap_values = pl.concat([male_shap_values, female_shap_values])

    grouped_shap_plot(shap_values=shap_values, column_mapping=column_mapping)
    # sex_shap_plot(df=shap_values, column_mapping=column_mapping)

    # color_mapping = dict(zip(names, sns.color_palette("tab20")))
    # shap_clustermap(
    #     shap_values=shap_values,
    #     feature_names=feature_names,
    #     column_mapping=column_mapping,
    #     color_mapping=color_mapping,
    # )
